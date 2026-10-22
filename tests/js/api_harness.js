// Runs static/common.js in Node with a fake fetch and prints what DS.api() throws.
// Usage: node api_harness.js <common.js> <i18n.json> <lang> <scenario>
const fs = require("fs");
const vm = require("vm");

const [commonPath, i18nPath, lang, scenario] = process.argv.slice(2);
const catalog = JSON.parse(fs.readFileSync(i18nPath, "utf8"));

globalThis.window = globalThis;
globalThis.document = {
  documentElement: { dataset: {}, classList: { remove() {} }, lang: "" },
  querySelectorAll: () => [],
};
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.location = { search: `?lang=${lang}`, href: "http://localhost/" };
globalThis.history = { replaceState() {} };

const scenarios = {
  network: () => Promise.reject(new TypeError("Failed to fetch")),
  timeout: () => Promise.reject(new DOMException("signal timed out", "TimeoutError")),
  not_json: () => Promise.resolve(new Response("<html>oops</html>", { status: 200 })),
  proxy_down: () => Promise.resolve(new Response("Bad Gateway", { status: 502 })),
  server_message: () => Promise.resolve(
    Response.json({ error: "not_found", message: "SERVER MESSAGE" }, { status: 404 }),
  ),
  ok: () => Promise.resolve(Response.json({ ok: true })),
};

globalThis.fetch = (url) => (url === "/static/i18n.json" ? Promise.resolve(Response.json(catalog)) : scenarios[scenario]());

vm.runInThisContext(fs.readFileSync(commonPath, "utf8"));

(async () => {
  await window.DS.loadTranslations();
  try {
    const body = await window.DS.api("/ask", { method: "POST" });
    console.log(JSON.stringify({ body }));
  } catch (e) {
    console.log(JSON.stringify({ message: e.message }));
  }
})();
