import AppKit
import Darwin
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// Drives the real composer/state/process/HTTP path against a small server, without model weights.
@MainActor
@Suite(.serialized)
struct ChatContextFlowTests {
    @Test func aFullChatKeepsTheDraftAndANewChatCanSendIt() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = """
                Summarize our decisions and next steps.
                Use the attached image to explain the
                interface changes, and recommend what
                we should implement next.
                """
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            let images = state.draft.images
            state.sendDraft()
            try await waitFor { !state.preparingToSend }
            let input = try #require(state.contextRequest)
            let usage = try #require(state.chatContext.usage(for: input))
            #expect(!usage.fits && !state.canSendDraft)
            #expect(state.draft.text == input.text && state.draft.images == images)
            #expect(state.chats.current?.messages == fixture.history.messages)
            #expect(try fixture.sentMessages().isEmpty)
            await picture("chat-context-full", state: state)
            await picture("chat-context-full-compact", state: state, width: 440)

            state.chats.newChat()
            state.draft.focus()
            let fresh = try #require(state.contextRequest)
            #expect(fresh.history.isEmpty && fresh.text == input.text && fresh.images == images)
            #expect(await state.chatContext.prepare(fresh)?.fits == true)
            await picture("chat-context-ready", state: state)
            state.sendDraft()
            try await waitFor { !state.preparingToSend && state.chats.streaming == nil }
            #expect(state.draft.isEmpty)
            #expect(state.chats.current?.messages.last?.text == "Answer")
            let sent = try fixture.sentMessages()
            #expect(sent.count == 1 && sent[0].count == 1)
            #expect(sent[0][0]["role"] as? String == "user")
            #expect(state.chats.conversations.first { $0.id == fixture.history.id } == fixture.history)
        }
    }

    @Test func editingDuringTheSendCheckCannotSendOrClearTheChangedDraft() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "slow question"
            state.sendDraft()
            try await waitFor { FileManager.default.fileExists(atPath: fixture.root.appending(path: "counting").path) }
            state.draft.text = "Changed question"
            try await waitFor { !state.preparingToSend }
            #expect(state.draft.text == "Changed question")
            #expect(state.chats.current?.messages == fixture.history.messages)
            #expect(try fixture.sentMessages().isEmpty)
        }
    }

    @Test func aCountFailureKeepsTheDraftAndCanRecover() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "offline"
            state.sendDraft()
            try await waitFor { !state.preparingToSend }
            let input = try #require(state.contextRequest)
            #expect(state.chatContext.failure(for: input) == "Try again shortly")
            #expect(state.draft.text == "offline")
            #expect(try fixture.sentMessages().isEmpty)
            state.draft.text = "Try now"
            state.sendDraft()
            try await waitFor { !state.preparingToSend && state.chats.streaming == nil }
            #expect(state.chats.current?.messages.last?.text == "Answer")
        }
    }

    private func withServer(_ test: (ContextFixture) async throws -> Void) async throws {
        let fixture = try ContextFixture()
        fixture.state.server.start("test-model", settings: ServeSettings(values: ["--port": String(fixture.port)]))
        do {
            try await waitFor { fixture.state.server.phase == .ready }
            try await test(fixture)
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }

    private func picture(_ name: String, state: AppState, width: CGFloat = 680) async {
        // As in MainView, give the composer a definite width before it measures the text field.
        await Snapshot.save(name, size: CGSize(width: width, height: 360), Offered {
            VStack {
                Spacer(minLength: 0)
                Composer(draft: state.draft).padding(16)
            }
        }.environment(state))
    }

    private func waitFor(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while !condition() {
            try #require(ContinuousClock.now < deadline, "The chat flow did not finish")
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

@MainActor
private final class ContextFixture {
    let root = FileManager.default.temporaryDirectory.appending(path: "splash-context-flow-\(UUID())")
    let history = Conversation(title: "Saved chat", messages: [ChatMessage(role: .user, text: String(repeating: "h", count: 600)),
                                                              ChatMessage(role: .assistant, text: "Saved answer")])
    let state: AppState
    let port: Int

    init() throws {
        let manager = FileManager.default
        for name in ["install", ".venv/bin", "chats"] {
            try manager.createDirectory(at: root.appending(path: name), withIntermediateDirectories: true)
        }
        // /usr/bin/python3 can be Apple's launcher; resolve the real interpreter before linking
        // it under the name `python`, which the launcher otherwise tries to find on PATH.
        let resolver = Process()
        resolver.executableURL = URL(filePath: "/usr/bin/python3")
        resolver.arguments = ["-c", "import sys; print(sys.executable)"]
        let output = Pipe()
        resolver.standardOutput = output
        try resolver.run()
        let executable = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        resolver.waitUntilExit()
        try #require(resolver.terminationStatus == 0 && !executable.isEmpty)
        try manager.createSymbolicLink(at: root.appending(path: ".venv/bin/python"), withDestinationURL: URL(filePath: executable))
        try Self.script.write(to: root.appending(path: "install/launcher.py"), atomically: true, encoding: .utf8)
        let chats = root.appending(path: "chats")
        try JSONEncoder().encode(history).write(to: chats.appending(path: "\(history.id).json"))
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        defer { Darwin.close(socket) }
        var address = sockaddr_in(sin_len: UInt8(MemoryLayout<sockaddr_in>.size), sin_family: sa_family_t(AF_INET),
                                  sin_port: 0, sin_addr: in_addr(s_addr: inet_addr("127.0.0.1")), sin_zero: (0, 0, 0, 0, 0, 0, 0, 0))
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = bind(socket, $0, length)
                _ = getsockname(socket, $0, &length)
            }
        }
        port = Int(UInt16(bigEndian: address.sin_port))
        let runtime = Runtime(root: root)
        state = AppState(runtime: runtime, chats: ChatStore(folder: chats), keychain: Keychain(service: "Splash Context Tests"),
                         server: Server(runtime: runtime, log: root.appending(path: "server.log")))
        state.chats.selection = history.id
    }

    func sentMessages() throws -> [[[String: Any]]] {
        let file = root.appending(path: "sent.jsonl")
        guard FileManager.default.fileExists(atPath: file.path) else { return [] }
        return try String(contentsOf: file, encoding: .utf8).split(separator: "\n").map {
            try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [[String: Any]])
        }
    }

    func close() async {
        state.chatContext.refresh(nil)
        state.server.stop()
        let deadline = ContinuousClock.now + .seconds(5)
        while state.server.running && ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(20))
        }
        try? FileManager.default.removeItem(at: root)
    }

    private static let script = #"""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    port = int(sys.argv[sys.argv.index('--port') + 1])
    lock = root / 'build/runtime' / f'serve-{port}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({'pid': os.getpid()}))
    def size(content):
        if isinstance(content, str): return len(content.encode())
        return sum(64 if part['type'] == 'image_url' else len(part['text'].encode()) for part in content)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, body, status=200, kind='application/json'):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path == '/v1/models':
                self.reply({'data': [{'context_length': 1024, 'input_modalities': ['text', 'image']}]})
            else: self.reply({})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/count_tokens'):
                text = body['messages'][-1]['content']
                if text == 'offline':
                    return self.reply({'error': {'message': 'Try again shortly'}}, 503)
                if text == 'slow question':
                    (root / 'counting').touch()
                    time.sleep(0.3)
                count = 100 + sum(size(m['content']) + len(m.get('reasoning_content', '').encode()) for m in body['messages'])
                return self.reply({'input_tokens': count})
            with (root / 'sent.jsonl').open('a') as output:
                output.write(json.dumps(body['messages']) + '\n')
            self.reply('data: {"choices":[{"delta":{"content":"Answer"}}]}\n\ndata: [DONE]\n\n', kind='text/event-stream')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
    """#
}
