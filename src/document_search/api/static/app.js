const { t, el, api } = window.DS;
const $ = (id) => document.getElementById(id);
let mode = "ask";
let lastOutput = null; // { kind, data } of the last successful query, re-rendered on language change

function message(kind, text) { return el("div", { class: `msg ${kind}` }, text); }

// --- status and document list --------------------------------------------------
async function refreshStatus() {
  try {
    const [stats, docs] = await Promise.all([api("/stats"), api("/documents")]);
    $("status").replaceChildren(
      el("span", { class: "pill" }, t("ui.status.documents", { n: stats.documents })),
      el("span", { class: "pill" }, t("ui.status.chunks", { n: stats.chunks })),
      el("span", { class: "pill" }, t("ui.status.index", { name: stats.index_backend })),
      el("span", { class: `pill ${stats.llm_enabled ? "on" : "off"}` },
        t(stats.llm_enabled ? "ui.status.llm_on" : "ui.status.llm_off")),
    );
    renderDocuments(docs);
  } catch (e) {
    $("status").replaceChildren(el("span", { class: "pill off" }, t("ui.status.offline")));
  }
}

function renderDocuments(docs) {
  if (!docs.length) {
    $("docs").replaceChildren(el("li", { class: "empty" }, t("ui.upload.empty")));
    return;
  }
  $("docs").replaceChildren(...docs.map((d) => {
    const label = t("ui.upload.delete", { name: d.filename });
    const remove = el("button", { class: "icon-btn", type: "button", title: label, "aria-label": label }, "×");
    remove.addEventListener("click", () => deleteDocument(d, remove));
    return el("li", {}, el("span", { class: "name" }, d.filename), " ",
      el("span", { class: "count" }, t("ui.upload.chunks_short", { n: d.chunks })), remove);
  }));
}

async function deleteDocument(doc, button) {
  if (!confirm(t("ui.upload.delete_confirm", { name: doc.filename }))) return;
  button.disabled = true;
  const box = $("uploadMsg");
  try {
    await api(`/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
    box.replaceChildren(message("ok", t("ui.upload.deleted", { name: doc.filename })));
    // Old results may cite the removed document.
    lastOutput = null;
    $("output").replaceChildren();
  } catch (e) {
    box.replaceChildren(message("err", `${doc.filename}: ${e.message}`));
    button.disabled = false;
  }
  refreshStatus();
}

// --- upload --------------------------------------------------------------------
async function uploadFiles(files) {
  const box = $("uploadMsg");
  box.replaceChildren();
  for (const file of files) {
    const line = message("warn", t("ui.upload.uploading", { name: file.name }));
    box.append(line);
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await api("/documents", { method: "POST", body: form });
      line.className = "msg ok";
      let text = t(r.duplicate ? "ui.upload.duplicate" : "ui.upload.indexed", { name: file.name, chunks: r.chunks });
      if (r.skipped_pages && r.skipped_pages.length) {
        text += "; " + t("ui.upload.skipped_pages", { pages: r.skipped_pages.join(", ") });
        line.className = "msg warn";
      }
      line.textContent = text;
    } catch (e) {
      line.className = "msg err";
      line.textContent = `${file.name}: ${e.message}`;
    }
  }
  refreshStatus();
}

// --- results -------------------------------------------------------------------
function sourceLocation(r) {
  return r.page != null ? t("ui.query.page", { n: r.page }) : t("ui.query.chunk", { n: r.chunk_index + 1 });
}

// Only the rank is shown: the raw similarity score reads like confidence, and it is not.
function resultCard(r, total) {
  return el("div", { class: "result", id: `src-${r.rank}` },
    el("div", { class: "meta" },
      el("span", { class: "num" }, `[${r.rank}]`),
      el("span", { class: "src" }, r.filename),
      el("span", {}, sourceLocation(r)),
      el("span", { class: "relevance", title: t("ui.query.rank_title") },
        t("ui.query.rank_label", { n: r.rank, total })),
    ),
    el("p", { class: "text" }, r.text),
  );
}

function jumpTo(n) {
  const target = $(`src-${n}`);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1200);
}

function answerWithLinks(text) {
  // Turn [n] markers into links to the matching source card; everything else stays plain text.
  const box = el("div", { class: "answer" });
  for (const part of text.split(/(\[\d+\])/)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) { box.append(part); continue; }
    const a = el("a", { href: `#src-${m[1]}` }, part);
    a.addEventListener("click", (ev) => { ev.preventDefault(); jumpTo(m[1]); });
    box.append(a);
  }
  return box;
}

function renderResults(out, results, labelKey) {
  if (!results.length) {
    out.append(message("warn", t("ui.query.no_results")));
    return;
  }
  out.append(el("div", { class: "label" }, t(labelKey)), ...results.map((r) => resultCard(r, results.length)));
}

// Collapsed by default: when there is no answer, the nearest passages are likely off-topic.
function closestPassages(results) {
  const box = el("details", { class: "closest" },
    el("summary", {}, t("ui.query.show_closest"), " ",
      el("span", { class: "caveat" }, `(${t("ui.query.may_be_unrelated")})`)));
  box.append(...results.map((r) => resultCard(r, results.length)));
  return box;
}

function renderOutput() {
  const out = $("output");
  out.replaceChildren();
  if (!lastOutput) return;
  const { kind, data: r } = lastOutput;
  if (kind === "search") {
    renderResults(out, r.results, "ui.query.nearest_label");
    return;
  }
  if (r.mode === "rag") {
    out.append(r.answer_found ? answerWithLinks(r.answer) : message("warn", t("ui.query.no_answer")));
    if (r.citations.length) {
      const links = el("p", { class: "hint sources" });
      r.citations.forEach((c, i) => {
        if (i) links.append(" · ");
        const a = el("a", { href: `#src-${c.rank}` }, `[${c.rank}] ${c.filename}, ${sourceLocation(c)}`);
        a.addEventListener("click", (ev) => { ev.preventDefault(); jumpTo(c.rank); });
        links.append(a);
      });
      out.append(el("div", { class: "label" }, t("ui.query.sources_label")), links);
    }
    if (!r.answer_found) {
      if (r.results.length) out.append(closestPassages(r.results));
      return;
    }
  } else if (r.notice) {
    // The server translated the notice for the language active when the request was sent.
    out.append(message("warn", t(`notices.${r.notice}`)));
  }
  renderResults(out, r.results, "ui.query.found_label");
}

async function runQuery(text) {
  const out = $("output");
  const button = $("go");
  button.disabled = true;
  out.replaceChildren(message("warn", t(mode === "ask" ? "ui.query.thinking" : "ui.query.searching")));
  try {
    const json = { method: "POST", headers: { "Content-Type": "application/json" } };
    lastOutput = mode === "search"
      ? { kind: "search", data: await api("/search", { ...json, body: JSON.stringify({ query: text, top_k: 5 }) }) }
      : { kind: "ask", data: await api("/ask", { ...json, body: JSON.stringify({ question: text, top_k: 5 }) }) };
    renderOutput();
  } catch (e) {
    lastOutput = null;
    out.replaceChildren(message("err", e.message));
  } finally {
    button.disabled = false;
  }
}

// --- mode tabs -----------------------------------------------------------------
function applyMode() {
  $("go").textContent = t(mode === "ask" ? "ui.query.button_ask" : "ui.query.button_search");
  $("hint").textContent = t(mode === "ask" ? "ui.query.hint_ask" : "ui.query.hint_search");
  $("query").placeholder = t(mode === "ask" ? "ui.query.placeholder_ask" : "ui.query.placeholder_search");
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((other) => other.classList.toggle("active", other === tab));
  mode = tab.dataset.mode;
  applyMode();
  lastOutput = null;
  $("output").replaceChildren();
}));

// --- events --------------------------------------------------------------------
$("file").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  uploadFiles([...e.dataTransfer.files]);
});

$("queryForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("query").value.trim();
  if (!text) { $("output").replaceChildren(message("err", t("ui.query.empty_query"))); return; }
  runQuery(text);
});

async function logout() {
  try { await api("/auth/logout", { method: "POST" }); } finally { location.href = "/login"; }
}

window.DS.onLanguageChange(() => {
  applyMode();
  renderOutput();
  $("uploadMsg").replaceChildren();
  refreshStatus();
});

(async () => {
  await window.DS.loadTranslations();
  const logoutButton = el("button", { class: "ghost", type: "button", "data-i18n": "ui.controls.logout" });
  logoutButton.addEventListener("click", logout);
  window.DS.mountControls($("controls"), logoutButton);
  applyMode();
  refreshStatus();
})();
