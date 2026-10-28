#include "model/DeepSeek41.hpp"

#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/WeightSource.hpp"

#include <bit>
#include <memory>
#include <string>

namespace splash::model {
namespace {

constexpr uint64_t kInt32Bytes = 4, kInt64Bytes = 8;

// A sublayer's mix, in the image's order (GgufImage.cpp mixImage): its
// projection's F32 rows, its scale and its base.
DeepSeek41MixWeights readMix(WeightFile &file, const DeepSeek41Layout &layout, const std::string &label) {
  DeepSeek41MixWeights mix;
  mix.fn = file.section(uint64_t{MHC_MIXES} * MHC_STREAMS * layout.hiddenSize * kFloat32Bytes, label + "-fn");
  mix.scale = file.section(3 * kFloat32Bytes, label + "-scale");
  mix.base = file.section(MHC_MIXES * kFloat32Bytes, label + "-base");
  return mix;
}

// A float projection (BF16 or F32 as the GGUF stores it) that writes fp32,
// as the compressor, the indexer and their scores read it.
ops::Projection floatProjection(WeightFile &file, const BlockTargetFormat &format, uint32_t outputSize,
                                uint32_t inputSize, std::string_view label) {
  ops::Projection projection = format.projection(file, outputSize, inputSize, label);
  projection.destination = ops::FloatOutput::Float32;
  return projection;
}

} // namespace

DeepSeek41LayerWeights readDeepSeek41Layer(WeightFile &file, const BlockTargetFormat &format,
                                           const DeepSeek41Layout &layout, const MoeExperts &experts, uint32_t index) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint32_t hidden = layout.hiddenSize, latent = layout.attentionHeadDimension;
  const auto has = [&](uint64_t layers) { return DeepSeek41Dimensions::has(layers, index); };
  DeepSeek41LayerWeights layer;
  if (has(d.engramLayers)) {
    DeepSeek41EngramWeights engram;
    engram.kv = format.projection(file, (MHC_STREAMS + 1) * hidden, d.engramWidth(), "engram-kv");
    engram.query = file.section(uint64_t{MHC_STREAMS} * hidden * kFloat32Bytes, "engram-query");
    engram.key = file.section(uint64_t{MHC_STREAMS} * hidden * kFloat32Bytes, "engram-key");
    layer.engram = std::move(engram);
  }
  layer.attentionMix = readMix(file, layout, "attention-mix");
  layer.attentionNorm = format.norm(file, hidden, "attention-norm");
  layer.queryA = format.projection(file, d.queryRank, hidden, "query-a");
  layer.queryNorm = format.norm(file, d.queryRank, "query-norm");
  layer.queryB = format.projection(file, layout.attentionWidth, d.queryRank, "query-b");
  layer.kv = format.projection(file, latent, hidden, "kv");
  layer.kvNorm = format.norm(file, latent, "kv-norm");
  layer.sinks = file.section(uint64_t{layout.attentionQueryHeads} * kFloat32Bytes, "attention-sinks");
  if (has(d.compressorLayers)) {
    DeepSeek41CompressorWeights compressor;
    compressor.kv = floatProjection(file, format, latent, hidden, "compressor-kv");
    if (has(d.pairedLayers)) compressor.gate = floatProjection(file, format, latent, hidden, "compressor-gate");
    compressor.norm = format.norm(file, latent, "compressor-norm");
    compressor.indexKey = floatProjection(file, format, d.indexerHeadDimension, latent, "index-key");
    compressor.indexKeyNorm = format.norm(file, d.indexerHeadDimension, "index-key-norm");
    layer.compressor = std::move(compressor);
  }
  if (has(d.indexerLayers)) {
    DeepSeek41IndexerWeights indexer;
    indexer.query = floatProjection(file, format, d.indexerHeads * d.indexerHeadDimension, d.queryRank,
                                    "indexer-query");
    indexer.weights = floatProjection(file, format, d.indexerHeads, hidden, "indexer-weights");
    layer.indexer = std::move(indexer);
  }
  const uint32_t groupWidth = d.groupWidth(layout.attentionWidth);
  for (uint32_t group = 0; group < d.outputGroups; ++group)
    layer.outputA.push_back(format.projection(file, d.outputRank, groupWidth, "output-a"));
  layer.outputB = format.projection(file, hidden, d.outputGroups * d.outputRank, "output-b");
  layer.ffnMix = readMix(file, layout, "ffn-mix");
  layer.ffnNorm = format.norm(file, hidden, "ffn-norm");
  layer.ffn = readMoeBlock(file, layout, experts, index);
  return layer;
}

void requireDeepSeek41Layout(const DeepSeek41Layout &layout) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint64_t layers = layout.layers >= 64 ? ~uint64_t{0} : (uint64_t{1} << layout.layers) - 1;
  const auto among = [](uint64_t bits, uint64_t within) { return !(bits & ~within); };
  // The first layer that attends compressed groups reads its own compressor's
  // and indexer's.
  const uint64_t first = d.compressedLayers & -d.compressedLayers;
  if (!d.present() || !layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      layout.attentionHeadDimension != LATENT_ATTENTION_DIM || layout.attentionKvHeads != 1 ||
      !layout.attentionQueryHeads || layout.attentionQueryHeads > LATENT_ATTENTION_MAX_HEADS ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      2 * layout.rotaryPairs > layout.attentionHeadDimension || !layout.rotaryPairs ||
      !d.outputRank || !d.outputGroups || layout.attentionWidth % d.outputGroups || !d.queryRank ||
      d.queryRank > LATENT_ROTATION_MAX_WIDTH || layout.hiddenSize > MHC_MAX_HIDDEN ||
      d.indexerHeadDimension != LATENT_INDEX_DIM || !d.indexerHeads || d.indexerHeads > LATENT_INDEX_MAX_HEADS ||
      !d.indexerTopGroups || d.engramNgram < 2 || !d.engramHeads || !d.engramHeadDimension ||
      !d.sinkhornIterations || layout.ffnKind != QwenFfnKind::SparseMoe ||
      layout.moeRouter != ops::MoeRouter::SqrtSoftplusNormalized || layout.sharedExpertSlices != 1 ||
      layout.sharedExpertGate || layout.leadingDenseLayers || !layout.experts || !layout.expertsPerToken ||
      layout.expertsPerToken > layout.experts || !layout.expertIntermediateSize ||
      !among(d.compressedLayers, layers) || !among(d.pairedLayers, d.compressedLayers) ||
      !among(d.compressorLayers, d.compressedLayers) || !among(d.indexerLayers, d.compressedLayers) ||
      !among(d.engramLayers, layers) || (first && (!(d.compressorLayers & first) || !(d.indexerLayers & first))))
    throw WeightStoreError("DeepSeek-V4.1 layout is inconsistent");
}

std::filesystem::path deepSeek41EngramPath(const GgufFile &target, const std::filesystem::path &targetPart) {
  const std::optional<std::string> name = target.metadata().stringValue("deepseek41.engram.file");
  if (!name || name->empty() || std::filesystem::path(*name).has_parent_path())
    throw WeightStoreError("the DeepSeek-V4.1 GGUF names no Engram file");
  return targetPart.parent_path().parent_path() / "engram" / *name;
}

DeepSeek41EngramTables deepSeek41EngramTables(const GgufFile &engram, uint64_t dataOffset,
                                              const DeepSeek41Layout &layout) {
  constexpr uint32_t kIq4Nl = 20; // GGML_TYPE_IQ4_NL
  const DeepSeek41Dimensions &d = layout.deepseek;
  DeepSeek41EngramTables result{.table = {.name = "engram", .rowBytes = d.engramHeadDimension / 32 * 18},
                                .format = GGUF_FMT_IQ4NL};
  uint64_t next = 0;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    if (!DeepSeek41Dimensions::has(d.engramLayers, layer)) continue;
    const GgufTensor &table = engram.require("blk." + std::to_string(layer) + ".engram_embd.weight");
    if (table.type != kIq4Nl || table.columns() != d.engramHeadDimension || table.file != 0 ||
        table.bytes != table.rows() * result.table.rowBytes)
      throw WeightStoreError("an Engram table is no IQ4_NL rows of the Engram's head width");
    if (result.firstRows.empty()) result.table.offset = dataOffset + table.offset;
    else if (table.offset != next) throw WeightStoreError("the Engram's tables do not lie back to back");
    next = table.offset + table.bytes;
    result.firstRows.push_back(static_cast<uint32_t>(result.table.rows));
    result.table.rows += table.rows();
  }
  result.table.validate();
  return result;
}

EngramHashTables DeepSeek41Weights::engramHash() const noexcept {
  const auto values = [](const metal::MetalBuffer &buffer, auto *type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return std::span(static_cast<const T *>(buffer.contents()), buffer.sizeBytes() / sizeof(T));
  };
  const DeepSeek41Dimensions &d = layout.deepseek;
  return {.tokenMap = values(engramTokenMap, static_cast<int32_t *>(nullptr)),
          .multipliers = values(engramMultipliers, static_cast<int64_t *>(nullptr)),
          .primes = values(engramPrimes, static_cast<int64_t *>(nullptr)),
          .offsets = values(engramOffsets, static_cast<int64_t *>(nullptr)),
          .layers = static_cast<uint32_t>(std::popcount(d.engramLayers)),
          .ngram = d.engramNgram,
          .heads = d.engramHeads,
          .padToken = d.engramPadToken};
}

DeepSeek41Weights loadDeepSeek41Weights(metal::MetalBackend &backend, DeepSeek41Layout layout,
                                        const QwenTargetFiles &files) {
  requireDeepSeek41Layout(layout);
  const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!gguf) throw WeightStoreError(std::string(layout.family) + " loads from a GGUF only");
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  const MoeExperts experts = openMoeExperts(backend, files, layout);
  const BlockTargetFormat format;
  GgufTargetLoader &loader = gguf->get();
  DeepSeek41Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    WeightFile file = loader.layer(index);
    result.layers.push_back(readDeepSeek41Layer(file, format, layout, experts, index));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    const DeepSeek41Dimensions &d = layout.deepseek;
    const uint64_t engrams = std::popcount(d.engramLayers), hashes = uint64_t{d.engramNgram - 1} * d.engramHeads;
    WeightFile file = loader.head();
    result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
    result.engramTokenMap = file.section(uint64_t{layout.vocabularySize} * kInt32Bytes, "engram-token-map");
    result.engramMultipliers = file.section(engrams * d.engramNgram * kInt64Bytes, "engram-multipliers");
    result.engramPrimes = file.section(engrams * hashes * kInt64Bytes, "engram-primes");
    result.engramOffsets = file.section(engrams * hashes * kInt64Bytes, "engram-offsets");
    // fp32 logits, as every target's (readQwenTargetHead).
    result.logitsProjection = floatProjection(file, format, layout.vocabularySize, layout.hiddenSize, "logits");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.embedding();
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }
  result.experts = experts.store;
  // The Engram's tables stream from their own file through the table's row
  // store, where the loader's options give it memory.
  if (const SparseRowStoreOptions &rows = loader.tableOptions(); rows.cacheBytes) {
    const auto source =
        std::make_shared<WeightSource>(deepSeek41EngramPath(loader.file(), loader.file().source().path()));
    const GgufFile engram(*source);
    const DeepSeek41EngramTables tables = deepSeek41EngramTables(engram, source->dataOffset(), layout);
    result.streamedTable = std::make_shared<SparseRowStore>(tables.table, openRowReader(source), rows);
    result.engramFirstRows = tables.firstRows;
    result.engramFormat = tables.format;
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
