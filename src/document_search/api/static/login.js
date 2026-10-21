const { t, api } = window.DS;

document.getElementById("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const box = document.getElementById("loginMsg");
  const button = event.target.querySelector("button");
  button.disabled = true;
  box.replaceChildren();
  try {
    await api("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: document.getElementById("password").value }),
    });
    location.href = "/";
  } catch (e) {
    box.className = "msg err";
    box.textContent = e.message || t("ui.login.failed");
    button.disabled = false;
  }
});

window.DS.onLanguageChange(() => document.getElementById("loginMsg").replaceChildren());

(async () => {
  await window.DS.loadTranslations();
  window.DS.mountControls(document.getElementById("controls"));
  document.getElementById("password").focus();
})();
