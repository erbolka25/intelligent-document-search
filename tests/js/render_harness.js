// Runs static/common.js and static/app.js on a tiny fake DOM, renders one /ask or /search
// response and prints the output tree as JSON.
// Usage: node render_harness.js <static dir> <lang> <kind> <response.json>
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const [staticDir, lang, kind, responsePath] = process.argv.slice(2);
const catalog = JSON.parse(fs.readFileSync(path.join(staticDir, "i18n.json"), "utf8"));

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.style = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name]; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener() {}
  querySelectorAll() { return []; }
  set textContent(value) { this.children = [String(value)]; }
  get textContent() {
    return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
  }
  toJSON() {
    return { tag: this.tag, class: this.className, attrs: this.attrs, children: this.children };
  }
}

const elements = {};
globalThis.window = globalThis;
globalThis.document = {
  documentElement: new FakeElement("html"),
  createElement: (tag) => new FakeElement(tag),
  getElementById: (id) => (elements[id] ||= new FakeElement("div")),
  querySelectorAll: () => [],
};
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.location = { search: `?lang=${lang}`, href: "http://localhost/" };
globalThis.history = { replaceState() {} };
globalThis.confirm = () => false;
globalThis.fetch = (url) => (url === "/static/i18n.json"
  ? Promise.resolve(Response.json(catalog))
  : Promise.reject(new TypeError("Failed to fetch")));

vm.runInThisContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(staticDir, "app.js"), "utf8"));

(async () => {
  await window.DS.loadTranslations();
  globalThis.__response = { kind, data: JSON.parse(fs.readFileSync(responsePath, "utf8")) };
  // lastOutput and renderOutput are top-level bindings of app.js, shared across scripts.
  vm.runInThisContext("lastOutput = __response; renderOutput();");
  console.log(JSON.stringify(document.getElementById("output")));
})();
