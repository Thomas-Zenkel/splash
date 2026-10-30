import Foundation
import Synchronization
import Testing

@testable import Splash

@MainActor
struct ChatContextTests {
    private func request(_ text: String = "Hello") -> ChatContext.Request {
        ChatContext.Request(history: [], text: text, images: [],
                            target: ChatTarget(url: URL(string: "http://127.0.0.1:9")!, model: "a/b",
                                               acceptsImages: true, effort: "high", context: 32768, apiKey: "secret"))
    }

    @Test func keepsSpaceForThinkingAndTheReply() {
        let full = ContextUsage(input: 28672, limit: 32768)
        #expect(full.reserved == 4096 && full.fits && full.nearLimit)
        #expect(!ContextUsage(input: 28673, limit: 32768).fits)
        #expect(ContextUsage(input: 896, limit: 1024).fits)
        #expect(!ContextUsage(input: 897, limit: 1024).fits)
        #expect(ContextUsage(input: 1, limit: 1).available == 0)
        let overflow = ContextUsage(input: 40000, limit: 32768)
        #expect(overflow.remaining == 0 && overflow.fraction == 1 && !overflow.fits)
        #expect(!ContextUsage(input: 100, limit: 32768).nearLimit)
    }

    @Test func countsTheSameImagesHistoryAndReasoningAsGeneration() throws {
        let history = [ChatMessage(role: .user, text: "你好", images: [Data([0x89, 0x50, 0x4E, 0x47])]),
                       ChatMessage(role: .assistant, text: "A cat", reasoning: "Whiskers"),
                       ChatMessage(role: .assistant, text: "", error: "Interrupted")]
        let input = ChatContext.Request(history: history, text: "Sure?", images: [], target: request().target)
        let count = try ChatClient.countRequest(input.messages, with: input.target)
        var generation = try #require(JSONSerialization.jsonObject(with: ChatClient.body(input.messages, effort: "high", images: true)) as? [String: Any])
        for key in ["stream", "stream_options", "return_progress"] { generation.removeValue(forKey: key) }
        let body = try #require(count.httpBody)
        let counted = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect((counted as NSDictionary).isEqual(to: generation))
        #expect(count.url?.path == "/v1/chat/completions/count_tokens")
        #expect(count.httpMethod == "POST")
        #expect(count.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
    }

    @Test func sendingSkipsTheTypingDelayAndReusesTheExactCount() async throws {
        let probe = CounterProbe()
        let context = ChatContext(counter: { try await probe.count($0) })
        let input = request()
        context.refresh(input)
        let usage = await context.prepare(input)
        #expect(usage?.input == input.text.utf8.count)
        #expect(await context.prepare(input) == usage)
        #expect(await probe.calls == 1)
        // Counts of another draft or model cannot enable Send.
        #expect(context.usage(for: request("Different")) == nil)
        var target = input.target
        target.context = 1024
        #expect(context.usage(for: .init(history: [], text: input.text, images: [], target: target)) == nil)
        target = ChatTarget(url: input.target.url, model: "a/other", acceptsImages: true, effort: "high", context: 32768)
        #expect(context.usage(for: .init(history: [], text: input.text, images: [], target: target)) == nil)
        #expect(context.usage(for: .init(history: [], text: input.text, images: [Data([1])], target: input.target)) == nil)
        #expect(context.usage(for: .init(history: [], text: input.text, images: [], target: input.target, started: .now)) == nil)
    }

    @Test func anOldCheckCannotReplaceANewerDraftsCount() async throws {
        let probe = SuspendedCounter()
        let context = ChatContext(counter: { await probe.count($0.text) })
        let old = request("old"), current = request("current")
        let first = Task { await context.prepare(old) }
        try await waitFor { await probe.started("old") }
        let second = Task { await context.prepare(current) }
        try await waitFor { await probe.started("current") }
        await probe.finish("current", with: 20)
        #expect(await second.value?.input == 20)
        // This counter deliberately ignores cancellation, as an already completed HTTP response
        // may arrive meanwhile. Its result must still be discarded.
        await probe.finish("old", with: 99999)
        #expect(await first.value == nil)
        #expect(context.usage(for: current)?.input == 20)
        context.refresh(nil)
        #expect(context.usage == nil && context.request == nil && context.failure == nil)
    }

    @Test func aFailedCheckCanBeRetried() async {
        let probe = CounterProbe(failFirst: true)
        let context = ChatContext(counter: { try await probe.count($0) })
        let input = request()
        #expect(await context.prepare(input) == nil)
        #expect(context.failure(for: input) == "Offline")
        #expect(context.failure(for: request("changed")) == nil)
        #expect(await context.prepare(input)?.fits == true)
        #expect(context.failure == nil)
    }

    @Test func validatesCountResponsesAndSurfacesServerErrors() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CountProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        for (status, payload, expected) in [
            (200, #"{"input_tokens":123}"#, Optional(123)),
            (200, #"{"input_tokens":-1}"#, nil),
            (200, "not json", nil),
            (503, #"{"error":{"message":"Please retry"}}"#, nil),
        ] {
            let url = URL(string: "http://\(UUID().uuidString).invalid")!
            let target = ChatTarget(url: url, model: "a/b", acceptsImages: false, effort: "")
            let endpoint = url.appending(path: "v1/chat/completions/count_tokens")
            CountProtocol.responses.withLock { $0[endpoint.absoluteString] = (status, Data(payload.utf8)) }
            defer { _ = CountProtocol.responses.withLock { $0.removeValue(forKey: endpoint.absoluteString) } }
            do {
                let result = try await ChatClient.countTokens([ChatMessage(role: .user, text: "Hi")], with: target, session: session)
                #expect(result == expected)
            } catch {
                #expect(expected == nil)
                #expect(error.localizedDescription == (status == 503 ? "Please retry" : "The model returned an invalid token count."))
            }
        }
    }

    @Test func callersOutsideTheComposerCannotGenerateAnOversizedPrompt() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-context-\(UUID())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let store = ChatStore(folder: folder)
        // The invalid local port would fail differently if generation were attempted.
        store.send("Keep this question", images: [], to: request().target, countedTokens: 32768)
        try await waitFor { @MainActor in store.streaming == nil }
        #expect(store.current?.messages.first?.text == "Keep this question")
        #expect(store.current?.messages.last?.error == ContextUsage(input: 32768, limit: 32768).problem)
    }

    private func waitFor(_ condition: () async -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while !(await condition()) {
            try #require(ContinuousClock.now < deadline, "Context check did not finish")
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}

private actor CounterProbe {
    private(set) var calls = 0
    let failFirst: Bool
    init(failFirst: Bool = false) { self.failFirst = failFirst }
    func count(_ request: ChatContext.Request) throws -> Int {
        calls += 1
        if failFirst && calls == 1 { throw ChatClient.Failure(errorDescription: "Offline") }
        return request.text.utf8.count
    }
}

private actor SuspendedCounter {
    private var pending: [String: CheckedContinuation<Int, Never>] = [:]
    func count(_ text: String) async -> Int {
        await withCheckedContinuation { pending[text] = $0 }
    }
    func started(_ text: String) -> Bool { pending[text] != nil }
    func finish(_ text: String, with count: Int) { pending.removeValue(forKey: text)?.resume(returning: count) }
}

private final class CountProtocol: URLProtocol, @unchecked Sendable {
    static let responses = Mutex<[String: (Int, Data)]>([:])
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url, let (status, data) = Self.responses.withLock({ $0[url.absoluteString] }),
              let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
