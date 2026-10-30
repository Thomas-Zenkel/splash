import Foundation
import Observation

/// The space the next prompt leaves in the running model's context. Thinking and the answer
/// share the remaining tokens; keep a useful minimum free rather than sending a prompt that
/// leaves room for only a few words. This is an allowance, not a cap on the reply's length.
struct ContextUsage: Equatable, Sendable {
    let input: Int
    let limit: Int

    var reserved: Int { min(4096, max(1, limit / 8)) }
    var available: Int { max(0, limit - reserved) }
    var remaining: Int { max(0, limit - input) }
    var fits: Bool { input <= available }
    var nearLimit: Bool { Double(input) >= Double(available) * 0.8 }
    var fraction: Double { min(1, max(0, Double(input) / Double(max(1, limit)))) }

    var explanation: String {
        "\(input.formatted()) of \(limit.formatted()) tokens used, including chat formatting and images. "
            + "At least \(reserved.formatted()) tokens are kept free for thinking and the reply."
    }

    var problem: String {
        "Not enough context for a reply. Shorten this message or start a new chat to leave "
            + "\(reserved.formatted()) tokens for thinking and the reply."
    }
}

/// Measures the exact prompt, after a short pause in typing. A result belongs to its whole input:
/// history, draft, images, model, reasoning effort and context. Editing or switching chats cannot
/// use an earlier result, even if an old HTTP request finishes after its cancellation.
@MainActor @Observable
final class ChatContext {
    struct Request: Equatable, Sendable {
        let history: [ChatMessage]
        let text: String
        let images: [Data]
        let target: ChatTarget
        /// A restarted model may use another tokenizer even under the same name.
        var started: Date? = nil

        var messages: [ChatMessage] {
            history + (text.isEmpty && images.isEmpty ? [] : [ChatMessage(role: .user, text: text, images: images)])
        }
    }

    typealias Counter = @Sendable (Request) async throws -> Int

    private(set) var request: Request?
    private(set) var usage: ContextUsage?
    private(set) var failure: String?
    @ObservationIgnored private let counter: Counter
    @ObservationIgnored private var task: Task<ContextUsage?, Never>?
    @ObservationIgnored private var revision = 0

    init(counter: @escaping Counter = { try await ChatClient.countTokens($0.messages, with: $0.target) }) {
        self.counter = counter
    }

    func usage(for request: Request) -> ContextUsage? {
        self.request == request ? usage : nil
    }

    func failure(for request: Request) -> String? {
        self.request == request ? failure : nil
    }

    func refresh(_ request: Request?) {
        guard self.request != request else { return }
        begin(request, delay: .milliseconds(300))
    }

    /// Send needn't wait for the typing delay. It checks immediately, or reuses a count of the
    /// exact same prompt; the draft is taken only after this succeeds.
    func prepare(_ request: Request) async -> ContextUsage? {
        if let usage = usage(for: request) { return usage }
        return await begin(request, delay: .zero)?.value
    }

    func retry(_ request: Request) {
        begin(request, delay: .zero)
    }

    @discardableResult
    private func begin(_ request: Request?, delay: Duration) -> Task<ContextUsage?, Never>? {
        task?.cancel()
        revision += 1
        let revision = revision
        self.request = request
        usage = nil
        failure = nil
        task = nil
        guard let request, let limit = request.target.context, limit > 0 else { return nil }
        let counter = counter
        let task = Task { [weak self] () -> ContextUsage? in
            do {
                try await Task.sleep(for: delay)
                let count = try await counter(request)
                try Task.checkCancellation()
                guard let self, self.revision == revision else { return nil }
                let usage = ContextUsage(input: count, limit: limit)
                self.usage = usage
                self.task = nil
                return usage
            } catch {
                guard !Task.isCancelled, let self, self.revision == revision else { return nil }
                self.failure = error.localizedDescription
                self.task = nil
                return nil
            }
        }
        self.task = task
        return task
    }
}
