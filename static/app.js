const form = document.querySelector("#predict-form");
const fileInput = document.querySelector("#file-input");
const fileName = document.querySelector("#file-name");
const dropzone = document.querySelector("#dropzone");
const modelList = document.querySelector("#model-list");
const statusBox = document.querySelector("#status");
const resultCount = document.querySelector("#result-count");
const summaryGrid = document.querySelector("#summary-grid");
const resultTable = document.querySelector("#result-table");
const downloadButton = document.querySelector("#download-button");

let lastResults = [];

const columns = [
  ["input_index", "Row"],
  ["alloy", "Alloy"],
  ["model", "Model"],
  ["predicted_tts_degC", "TTS degC"],
  ["uncertainty_1sigma_degC", "1 sigma"],
  ["lower_95_degC", "95% low"],
  ["upper_95_degC", "95% high"],
  ["domain_status", "Domain"],
  ["domain_outside_count", "OOD fields"],
  ["domain_outside_features", "Domain details"],
  ["temperature_C", "Temp C"],
  ["fluence_n_cm2", "Fluence"],
  ["flux_n_cm2_sec", "Flux"],
  ["Product Form", "Product Form"],
  ["Reactor Type", "Reactor"],
  ["wt_percent_Cu", "Cu wt%"],
  ["wt_percent_Ni", "Ni wt%"],
  ["wt_percent_Mn", "Mn wt%"],
  ["wt_percent_P", "P wt%"],
  ["wt_percent_Si", "Si wt%"],
  ["wt_percent_C", "C wt%"]
];

async function init() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    renderModels(data.models);
  } catch (error) {
    setStatus("Could not load model metadata.", true);
  }
}

function renderModels(models) {
  modelList.innerHTML = "";
  models.forEach((model, index) => {
    const label = document.createElement("label");
    label.className = "model-option";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "models";
    checkbox.value = model.key;
    checkbox.checked = index < 2;

    const text = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = model.label;
    const sub = document.createElement("small");
    sub.textContent = `${model.benchmark.n} benchmark residuals`;
    text.append(name, sub);

    const pill = document.createElement("span");
    pill.className = "metric-pill";
    pill.textContent = `sigma ${formatNumber(model.benchmark.residual_std_degC)}`;

    label.append(checkbox, text, pill);
    modelList.append(label);
  });
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileName.textContent = file ? file.name : "Choose input file";
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  fileInput.files = event.dataTransfer.files;
  fileName.textContent = file.name;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  const selected = Array.from(document.querySelectorAll("input[name='models']:checked")).map((item) => item.value);

  if (!file) {
    setStatus("Choose an input file.", true);
    return;
  }
  if (selected.length === 0) {
    setStatus("Choose at least one model.", true);
    return;
  }

  const body = new FormData();
  body.append("file", file);
  body.append("models", selected.join(","));

  setStatus("Running predictions...", false);
  form.querySelector("button[type='submit']").disabled = true;

  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    lastResults = data.results || [];
    renderSummary(data);
    renderTable(lastResults);
    downloadButton.disabled = lastResults.length === 0;
    const warningText = data.warnings && data.warnings.length ? ` ${data.warnings.join(" ")}` : "";
    setStatus(`Finished ${data.result_rows} result rows.${warningText}`, false, true);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    form.querySelector("button[type='submit']").disabled = false;
  }
});

downloadButton.addEventListener("click", () => {
  if (!lastResults.length) return;
  const csv = toCsv(lastResults);
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "rpv_predictions.csv";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
});

function renderSummary(data) {
  summaryGrid.innerHTML = "";
  const cards = [
    ["Input rows", data.input_rows],
    ["Result rows", data.result_rows],
    ["Models", data.models.join(", ")]
  ];

  cards.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    const small = document.createElement("small");
    small.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    card.append(small, strong);
    summaryGrid.append(card);
  });
}

function renderTable(rows) {
  resultTable.querySelector("thead").innerHTML = "";
  resultTable.querySelector("tbody").innerHTML = "";
  resultCount.textContent = rows.length ? `${rows.length} rows` : "No results";

  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    const td = document.createElement("td");
    td.textContent = "Upload a file to populate predictions.";
    tr.append(td);
    resultTable.querySelector("tbody").append(tr);
    return;
  }

  const headerRow = document.createElement("tr");
  columns.forEach(([, label]) => {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.append(th);
  });
  resultTable.querySelector("thead").append(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach(([key]) => {
      const td = document.createElement("td");
      td.textContent = row[key] ?? "";
      tr.append(td);
    });
    resultTable.querySelector("tbody").append(tr);
  });
}

function setStatus(message, isError = false, isOk = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
  statusBox.classList.toggle("ok", isOk);
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(1);
}

function toCsv(rows) {
  const header = columns.map(([, label]) => label);
  const lines = [header.join(",")];
  rows.forEach((row) => {
    lines.push(columns.map(([key]) => csvEscape(row[key] ?? "")).join(","));
  });
  return `${lines.join("\n")}\n`;
}

function csvEscape(value) {
  const text = String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

init();
