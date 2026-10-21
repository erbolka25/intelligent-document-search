// Shared by the main and login pages: theme, translations, API calls and header controls.
// Loaded in <head> without defer, so the saved theme applies before the first paint.
(() => {
  const DEFAULT_LANG = "ru";
  const THEMES = ["auto", "light", "dark"];

  const store = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage blocked */ } },
  };

  // --- theme ------------------------------------------------------------------
  function currentTheme() {
    const saved = store.get("theme");
    return THEMES.includes(saved) ? saved : "auto";
  }

  function applyTheme(theme) {
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
  }

  applyTheme(currentTheme());

  // --- translations -----------------------------------------------------------
  let catalog = {};
  let lang = DEFAULT_LANG;
  const listeners = [];

  function lookup(tree, key) {
    return key.split(".").reduce((node, part) => (node && typeof node === "object" ? node[part] : undefined), tree);
  }

  function t(key, params = {}) {
    let text = lookup(catalog[lang], key);
    if (typeof text !== "string") text = lookup(catalog[DEFAULT_LANG], key);
    if (typeof text !== "string") return key;
    return text.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
  }

  function applyTranslations(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
    for (const attr of ["placeholder", "title", "aria-label"]) {
      root.querySelectorAll(`[data-i18n-${attr}]`).forEach((node) => {
        node.setAttribute(attr, t(node.getAttribute(`data-i18n-${attr}`)));
      });
    }
    document.documentElement.lang = lang;
  }

  async function loadTranslations() {
    try {
      catalog = await (await fetch("/static/i18n.json")).json();
    } catch {
      catalog = {};
    }
    const wanted = new URLSearchParams(location.search).get("lang") || store.get("lang");
    lang = wanted && wanted in catalog ? wanted : DEFAULT_LANG;
    applyTranslations();
    document.documentElement.classList.remove("i18n-pending");
  }

  function setLanguage(code) {
    if (!(code in catalog)) return;
    lang = code;
    store.set("lang", code);
    const url = new URL(location.href);
    if (url.searchParams.has("lang")) {
      url.searchParams.delete("lang");
      history.replaceState(null, "", url);
    }
    applyTranslations();
    listeners.forEach((fn) => fn(code));
  }

  // --- DOM and API helpers ------------------------------------------------------
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else node.setAttribute(k, v);
    }
    for (const c of children) node.append(c);
    return node;
  }

  async function api(path, options = {}) {
    const headers = { "X-Lang": lang, ...(options.headers || {}) };
    const resp = await fetch(path, { ...options, headers });
    if (resp.status === 401 && !path.startsWith("/auth/")) {
      location.href = "/login";
      throw new Error("unauthorized");
    }
    let body = null;
    try { body = await resp.json(); } catch { /* non-JSON body */ }
    if (!resp.ok) throw new Error((body && body.message) || t("ui.query.error_status", { status: resp.status }));
    return body;
  }

  // Language and theme selects, plus an optional extra control (the sign-out button).
  function mountControls(container, ...extra) {
    const langSelect = el("select", { "data-i18n-aria-label": "ui.controls.language" });
    for (const code of Object.keys(catalog)) {
      const name = (catalog[code].meta && catalog[code].meta.name) || code;
      langSelect.append(el("option", { value: code }, name));
    }
    langSelect.value = lang;
    langSelect.addEventListener("change", () => setLanguage(langSelect.value));

    const themeSelect = el("select", { "data-i18n-aria-label": "ui.controls.theme" });
    for (const theme of THEMES) {
      themeSelect.append(el("option", { value: theme, "data-i18n": `ui.controls.theme_${theme}` }));
    }
    themeSelect.value = currentTheme();
    themeSelect.addEventListener("change", () => {
      store.set("theme", themeSelect.value);
      applyTheme(themeSelect.value);
    });

    container.replaceChildren(langSelect, themeSelect, ...extra);
    applyTranslations(container);
  }

  window.DS = {
    t,
    el,
    api,
    applyTranslations,
    loadTranslations,
    mountControls,
    lang: () => lang,
    onLanguageChange: (fn) => listeners.push(fn),
  };
})();
