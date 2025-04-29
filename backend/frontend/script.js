const socket = new WebSocket("ws://localhost:8000/ws/guest123");

socket.onmessage = function (event) {
    if (event.data === "__typing__") {
        showTypingIndicator(true);
    } else {
        showTypingIndicator(false);
        appendMessage("bot", event.data);
    }
};

document.getElementById("send-btn").addEventListener("click", function () {
    const input = document.getElementById("user-input");
    const text = input.value.trim();
    if (text) {
        appendMessage("user", text);
        socket.send(text);
        input.value = "";
    }
});

function appendMessage(sender, text) {
    const container = document.getElementById("chat-container");
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.innerHTML = sender === "bot" 
        ? `<strong>Alex (Lodestone):</strong> ${text}`
        : `<strong>You:</strong> ${text}`;
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

function showTypingIndicator(show) {
    const typing = document.getElementById("typing-indicator");
    typing.style.display = show ? "block" : "none";
}

