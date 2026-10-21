const $ = (id) => document.getElementById(id);
let mode = "ask";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function message(kind, text) { return el("div", { class: `msg ${kind}` }, text); }

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (resp.status === 401 && !path.startsWith("/auth/")) {
    location.href = "/login";
    throw new Error("unauthorized");
  }
  let body = null;
  try { body = await resp.json(); } catch { /* non-JSON body */ }
  if (!resp.ok) {
    let detail = body && body.message;
    if (Array.isArray(detail)) detail = detail.map((d) => `${d.field}: ${d.message}`).join("; ");
    throw new Error(detail || `Ошибка ${resp.status}`);
  }
  return body;
}

async function refreshStatus() {
  try {
    const [h, docs] = await Promise.all([api("/stats"), api("/documents")]);
    $("status").replaceChildren(
      el("span", { class: "pill" }, `документов: ${h.documents}`),
      el("span", { class: "pill" }, `фрагментов: ${h.chunks}`),
      el("span", { class: "pill" }, `индекс: ${h.index_backend}`),
      el("span", { class: `pill ${h.llm_enabled ? "on" : "off"}` },
        h.llm_enabled ? "LLM подключена" : "LLM выключена: только поиск"),
    );
    $("docs").replaceChildren(...docs.map((d) =>
      el("li", {}, d.filename, " ", el("span", {}, `· ${d.chunks} фр.`))));
  } catch (e) {
    $("status").replaceChildren(el("span", { class: "pill off" }, "сервер недоступен"));
  }
}

async function uploadFiles(files) {
  const box = $("uploadMsg");
  box.replaceChildren();
  for (const file of files) {
    const line = message("warn", `Загружаю ${file.name}…`);
    box.append(line);
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await api("/documents", { method: "POST", body: form });
      line.className = "msg ok";
      line.textContent = r.duplicate
        ? `${file.name}: уже в индексе (${r.chunks} фр.)`
        : `${file.name}: проиндексирован, ${r.chunks} фр.`;
    } catch (e) {
      line.className = "msg err";
      line.textContent = `${file.name}: ${e.message}`;
    }
  }
  refreshStatus();
}

function sourceLocation(r) {
  return r.page != null ? `стр. ${r.page}` : `фрагмент ${r.chunk_index + 1}`;
}

function resultCard(r) {
  const pct = Math.max(0, Math.min(100, Math.round(r.score * 100)));
  return el("div", { class: "result", id: `src-${r.rank}` },
    el("div", { class: "meta" },
      el("span", { class: "num" }, `[${r.rank}]`),
      el("span", { class: "src" }, r.filename),
      el("span", {}, sourceLocation(r)),
      el("span", { class: "score", title: "косинусное сходство" },
        el("span", { class: "bar" }, el("i", { style: `width:${pct}%` })), r.score.toFixed(3)),
    ),
    el("p", { class: "text" }, r.text),
  );
}

function answerWithLinks(text) {
  // Turn [n] markers into links to the matching source card; everything else stays plain text.
  const box = el("div", { class: "answer" });
  for (const part of text.split(/(\[\d+\])/)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) { box.append(part); continue; }
    const a = el("a", { href: `#src-${m[1]}` }, part);
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const target = $(`src-${m[1]}`);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1200);
    });
    box.append(a);
  }
  return box;
}

function renderResults(out, results, label) {
  if (!results.length) {
    out.append(message("warn", "Ничего не найдено. Загрузите документы или переформулируйте запрос."));
    return;
  }
  out.append(el("div", { class: "label" }, label), ...results.map(resultCard));
}

async function runQuery(text) {
  const out = $("output");
  const button = $("go");
  button.disabled = true;
  out.replaceChildren(message("warn", mode === "ask" ? "Ищу ответ…" : "Ищу…"));
  try {
    if (mode === "search") {
      const r = await api("/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, top_k: 5 }),
      });
      out.replaceChildren();
      renderResults(out, r.results, "Наиболее близкие фрагменты");
    } else {
      const r = await api("/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, top_k: 5 }),
      });
      out.replaceChildren();
      if (r.mode === "rag") {
        out.append(r.answer_found
          ? answerWithLinks(r.answer)
          : message("warn", "В документах нет ответа на этот вопрос."));
        if (r.citations.length) {
          out.append(el("div", { class: "label" }, "Источники ответа"),
            el("p", { class: "hint" }, r.citations.map((c) => `[${c.rank}] ${c.filename}, ${sourceLocation(c)}`).join(" · ")));
        }
      } else if (r.notice) {
        out.append(message("warn", r.notice_message || r.notice));
      }
      renderResults(out, r.results, "Найденные фрагменты");
    }
  } catch (e) {
    out.replaceChildren(message("err", e.message));
  } finally {
    button.disabled = false;
  }
}

$("file").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  uploadFiles([...e.dataTransfer.files]);
});

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  mode = tab.dataset.mode;
  $("go").textContent = mode === "ask" ? "Спросить" : "Найти";
  $("hint").textContent = mode === "ask"
    ? "Ответ строится только по загруженным документам, с указанием источников."
    : "Поиск по смыслу, а не по точному совпадению слов.";
  $("output").replaceChildren();
}));

$("queryForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("query").value.trim();
  if (!text) { $("output").replaceChildren(message("err", "Введите запрос.")); return; }
  runQuery(text);
});

refreshStatus();
