// Session is kept in memory for the page; "New session" resets it.
let sessionId = null;

const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const modelSelect = document.getElementById("model-select");
const newSessionBtn = document.getElementById("new-session-btn");
const sendBtn = chatForm.querySelector("button[type=submit]");

function appendMessage(text, role) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    appendMessage(question, "user");
    chatInput.value = "";
    sendBtn.disabled = true;

    try {
        const payload = {
            question,
            model: modelSelect.value,
        };
        if (sessionId) payload.session_id = sessionId;

        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.text();
            appendMessage(`Error: ${err}`, "error");
            return;
        }

        const data = await res.json();
        sessionId = data.session_id;
        appendMessage(data.answer, "ai");
    } catch (err) {
        appendMessage(`Network error: ${err.message}`, "error");
    } finally {
        sendBtn.disabled = false;
        chatInput.focus();
    }
});

newSessionBtn.addEventListener("click", () => {
    sessionId = null;
    chatBox.innerHTML = "";
    chatInput.focus();
});