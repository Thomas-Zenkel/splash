// How an expert store's size lays its slots out in chunks (model::
// streamedExpertChunkLayout and wholeExpertChunkLayout), at two layouts: one
// whose planes each fill whole regions, and one whose slots fill them only
// eight at a time. Every layout takes at most MOE_EXPERT_MAX_CHUNKS chunks of
// at most the largest buffer, every chunk but the last of a multiple of the
// slots that fill whole regions, so the chunks take the bytes of one chunk of
// all the slots. A streamed size's floor is kExpertFloorShare of its slots
// rounded down to such a multiple, at least its minimum and at most the
// largest buffer; its elastic chunks take about the elastic bytes asked for
// (kExpertElasticChunkBytes by default), or more where the slots above the
// floor would otherwise take more than MOE_EXPERT_MAX_CHUNKS - 1 chunks, and
// a largest buffer that holds the size only in more than
// MOE_EXPERT_MAX_CHUNKS chunks is refused, the smallest that holds it in at
// most that many is not. A size of one block is its floor chunk alone,
// refused when the largest buffer does not hold it.
#include "TestChecks.hpp"
#include "model/ExpertChunkLayout.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using splash::model::alignWeightOffset;
using splash::model::ExpertChunkLayout;
using splash::model::ExpertSlotPlanes;
using splash::model::kExpertElasticChunkBytes;
using splash::model::kExpertFloorShare;
using splash::model::kWeightFileAlignment;
using splash::model::streamedExpertChunkLayout;
using splash::model::wholeExpertChunkLayout;
using splash::test::rejects;
using splash::test::require;

namespace {

// The 35B's Q4_K experts, 512 x 2048 gate and up and 2048 x 512 down: every
// plane fills whole regions.
constexpr ExpertSlotPlanes kWhole{{{524288, 0, 65536}, {524288, 0, 65536}, {524288, 0, 65536}}};
// Q5_K gate and up of 640 x 256 and a Q8_0 down of 256 x 640: plane1 fills
// whole regions four slots at a time and meta eight at a time.
constexpr ExpertSlotPlanes kEighths{{{81920, 20480, 10240}, {81920, 20480, 10240}, {163840, 0, 10240}}};
constexpr std::pair<ExpertSlotPlanes, const char *> kLayouts[] = {{kWhole, "whole"}, {kEighths, "eighths"}};

// The fewest slots whose every plane fills whole regions.
uint64_t alignedSlotsOf(const ExpertSlotPlanes &planes) {
  for (uint64_t slots = 1;; ++slots) {
    bool whole = true;
    for (const auto &projection : planes)
      for (const uint64_t plane : projection) whole = whole && slots * plane % kWeightFileAlignment == 0;
    if (whole) return slots;
  }
}

uint64_t alignDown(uint64_t value, uint64_t alignment) { return value / alignment * alignment; }
uint64_t alignUp(uint64_t value, uint64_t alignment) { return alignDown(value + alignment - 1, alignment); }

// The bytes of a chunk of `slots` slots: each plane in a region of its own.
uint64_t chunkBytes(const ExpertSlotPlanes &planes, uint64_t slots) {
  uint64_t bytes = 0;
  for (const auto &projection : planes)
    for (const uint64_t plane : projection) bytes += alignWeightOffset(slots * plane);
  return bytes;
}

// A layout of `slots` slots, which must take 1 to MOE_EXPERT_MAX_CHUNKS
// chunks, each of at most `largest` bytes and of at least one slot, the floor
// first and then elastic chunks of chunkSlots slots, the last of the rest;
// every chunk but the last of a multiple of the aligned slots, together of
// the bytes of one chunk of all the slots.
ExpertChunkLayout checked(const ExpertChunkLayout &layout, const ExpertSlotPlanes &planes, uint32_t slots,
                          uint64_t largest, const std::string &what) {
  require(layout.slots == slots && layout.floorSlots > 0 && layout.chunkSlots > 0 && layout.chunks >= 1 &&
              layout.chunks <= MOE_EXPERT_MAX_CHUNKS,
          what + ": " + std::to_string(layout.chunks) + " chunks, or a floor or elastic chunks of no slots");
  require(layout.chunks > 1 || (layout.floorSlots == slots && layout.chunkSlots == slots),
          what + ": a size of one chunk has other floor or chunk slots than its own");
  const uint64_t group = alignedSlotsOf(planes);
  uint64_t held = 0, bytes = 0;
  for (uint32_t chunk = 0; chunk < layout.chunks; ++chunk) {
    const uint64_t slotsOfChunk = layout.held(chunk);
    require(slotsOfChunk > 0 && slotsOfChunk <= (chunk ? layout.chunkSlots : layout.floorSlots),
            what + ": chunk " + std::to_string(chunk) + " holds no slots or more than its share");
    require(chunk + 1 == layout.chunks || slotsOfChunk % group == 0,
            what + ": chunk " + std::to_string(chunk) + " of " + std::to_string(slotsOfChunk) +
                " slots does not fill whole regions");
    require(chunkBytes(planes, slotsOfChunk) <= largest,
            what + ": chunk " + std::to_string(chunk) + " takes more than the largest buffer");
    held += slotsOfChunk;
    bytes += chunkBytes(planes, slotsOfChunk);
  }
  require(held == slots, what + ": the chunks hold " + std::to_string(held) + " slots");
  require(bytes == chunkBytes(planes, slots), what + ": the chunks take other bytes than one chunk of their slots");
  return layout;
}

ExpertChunkLayout streamed(const ExpertSlotPlanes &planes, uint32_t slots, uint32_t minimum, uint64_t largest,
                           const std::string &what, uint64_t elasticBytes = kExpertElasticChunkBytes) {
  return checked(streamedExpertChunkLayout(planes, slots, minimum, largest, elasticBytes), planes, slots, largest,
                 what);
}

// A large size's floor is the floor share of its slots rounded down to whole
// regions, the rest in elastic chunks of about kExpertElasticChunkBytes, the
// last of fewer: the 35B's experts in about 13.5 GiB, and the eighths in
// about 12 GiB.
void floorShareAndElasticChunks() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = planes == kWhole ? 8193 : 32771;
    const uint64_t group = alignedSlotsOf(planes);
    const ExpertChunkLayout layout = streamed(planes, slots, 512, UINT64_MAX, name);
    require(layout.floorSlots == alignDown(static_cast<uint64_t>(kExpertFloorShare * slots), group),
            std::string(name) + ": the floor is not the floor share rounded down to whole regions");
    require(chunkBytes(planes, layout.chunkSlots) <= kExpertElasticChunkBytes &&
                chunkBytes(planes, layout.chunkSlots + group) > kExpertElasticChunkBytes && layout.chunks >= 3 &&
                (slots - layout.floorSlots) % layout.chunkSlots,
            std::string(name) + ": the elastic chunks take other bytes than about the target, or the last is whole");
  }
}

// Elastic chunks take about the bytes asked for, beside a floor of the floor
// share: asked for the bytes of a sixteenth of its slots, a size takes a
// floor of the floor share and three elastic chunks of a sixteenth of its
// slots each, the last of fewer.
void elasticChunkBytes() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = 4099;
    const uint64_t group = alignedSlotsOf(planes);
    const ExpertChunkLayout layout =
        streamed(planes, slots, 512, UINT64_MAX, name, chunkBytes(planes, alignUp(slots / 16, group)));
    require(layout.floorSlots == alignDown(static_cast<uint64_t>(kExpertFloorShare * slots), group) &&
                layout.chunkSlots == alignUp(slots / 16, group) && layout.chunks == 4 &&
                (slots - layout.floorSlots) % layout.chunkSlots,
            std::string(name) + ": elastic chunks of a sixteenth of the slots are not what they hold, or the last "
                                "is whole");
  }
}

// A floor holds at least the size's minimum, rounded up to whole regions,
// and at most the slots; a size of fewer slots than fill whole regions is
// one chunk of them.
void floorMinimum() {
  for (const auto &[planes, name] : kLayouts) {
    const uint64_t group = alignedSlotsOf(planes);
    const ExpertChunkLayout raised = streamed(planes, 4096, 3999, UINT64_MAX, name);
    require(raised.floorSlots == alignUp(3999, group) && raised.chunks == 2 &&
                raised.chunkSlots == 4096 - raised.floorSlots,
            std::string(name) + ": a floor below the minimum, or the rest not in one elastic chunk");
    const ExpertChunkLayout all = streamed(planes, 515, 515, UINT64_MAX, name);
    require(all.chunks == 1, std::string(name) + ": a size of its minimum's slots takes elastic chunks");
  }
  require(streamed(kEighths, 3, 1, UINT64_MAX, "3 slots").chunks == 1 &&
              streamed(kWhole, 16, 16, UINT64_MAX, "16 slots").chunks == 1,
          "a size of fewer slots than a floor's group or minimum takes more than one chunk");
}

// A floor and elastic chunks take at most the largest buffer, and as many
// whole regions as it holds: under a largest buffer below the floor share and
// kExpertElasticChunkBytes, the floor and each elastic chunk but the last are
// what it holds, the floor also below the size's minimum.
void floorCap() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = 4096;
    const uint64_t group = alignedSlotsOf(planes);
    const uint64_t largest = chunkBytes(planes, 300) + 1;
    const ExpertChunkLayout capped = streamed(planes, slots, 2048, largest, name);
    require(capped.floorSlots == alignDown(300, group) && capped.chunkSlots == capped.floorSlots,
            std::string(name) + ": a floor or elastic chunk under a largest buffer takes less than it holds");
  }
}

// The slots above a floor that chunks of about kExpertElasticChunkBytes hold
// only in more than 47 take larger chunks, 47 at most: the 35B's experts in
// about 330 GiB.
void largerElasticChunks() {
  const uint32_t slots = 200000;
  const ExpertChunkLayout layout = streamed(kWhole, slots, 512, UINT64_MAX, "330 GiB");
  const uint64_t rest = slots - layout.floorSlots;
  require(chunkBytes(kWhole, layout.chunkSlots) > kExpertElasticChunkBytes &&
              layout.chunkSlots == (rest + MOE_EXPERT_MAX_CHUNKS - 2) / (MOE_EXPERT_MAX_CHUNKS - 1) &&
              layout.chunks == MOE_EXPERT_MAX_CHUNKS,
          "330 GiB: the elastic chunks are not the fewest slots that take 47 chunks");
}

// The smallest largest buffer that holds the size in at most 48 chunks is
// taken, a byte less refused, and one of a sixtieth of the size refused. The
// 20000-slot sizes take exactly 48 chunks under it; the 515-slot ones fewer.
void tooManyChunks() {
  for (const auto &[planes, name] : kLayouts)
    for (const uint32_t slots : {515u, 20000u}) {
      const std::string what = std::string(name) + ", " + std::to_string(slots) + " slots";
      const uint64_t fewest = alignUp((slots + MOE_EXPERT_MAX_CHUNKS - 1) / MOE_EXPERT_MAX_CHUNKS,
                                      alignedSlotsOf(planes));
      const uint64_t largest = chunkBytes(planes, fewest);
      const ExpertChunkLayout layout = streamed(planes, slots, 1, largest, what);
      require(layout.floorSlots == fewest && layout.chunkSlots == fewest &&
                  layout.chunks == (slots + fewest - 1) / fewest &&
                  (slots != 20000 || layout.chunks == MOE_EXPERT_MAX_CHUNKS),
              what + ": the floor and elastic chunks under the smallest largest buffer of at most 48 chunks are "
                     "not what it holds");
      const std::string expected = "takes more than " + std::to_string(MOE_EXPERT_MAX_CHUNKS) +
                                   " of the device's largest buffers";
      const auto under = [&](uint64_t buffer) {
        static_cast<void>(streamedExpertChunkLayout(planes, slots, 1, buffer, kExpertElasticChunkBytes));
      };
      rejects([&] { under(largest - 1); }, expected,
              what + ": a largest buffer a byte short of the smallest of at most 48 chunks was taken");
      rejects([&] { under(chunkBytes(planes, slots) / 60); }, expected,
              what + ": a largest buffer of a sixtieth of the size was taken");
    }
}

// A size of one block keeps every expert in its floor chunk alone, taken
// under a largest buffer of its bytes exactly and refused a byte less.
void oneBlockSizes() {
  for (const auto &[planes, name] : kLayouts)
    for (const uint32_t slots : {3u, 259u, 512u}) {
      const std::string what = std::string(name) + ", one block of " + std::to_string(slots) + " slots";
      const uint64_t bytes = chunkBytes(planes, slots);
      const ExpertChunkLayout layout =
          checked(wholeExpertChunkLayout(planes, slots, bytes), planes, slots, bytes, what);
      require(layout.chunks == 1, what + ": takes more than its floor chunk");
      rejects([&] { static_cast<void>(wholeExpertChunkLayout(planes, slots, bytes - 1)); },
              "more than the device's largest buffer", what + ": taken under a largest buffer a byte short of it");
    }
}

} // namespace

int main() {
  try {
    floorShareAndElasticChunks();
    elasticChunkBytes();
    floorMinimum();
    floorCap();
    largerElasticChunks();
    tooManyChunks();
    oneBlockSizes();
  } catch (const std::exception &error) {
    std::cerr << "expert_chunk_layout_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "expert_chunk_layout_test: PASS\n";
  return 0;
}
