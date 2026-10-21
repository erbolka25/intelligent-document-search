document.getElementById("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const box = document.getElementById("loginMsg");
  const resp = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password: document.getElementById("password").value }),
  });
  if (resp.ok) { location.href = "/"; return; }
  const body = await resp.json().catch(() => ({}));
  box.className = "msg err";
  box.textContent = body.message || "Error";
});
