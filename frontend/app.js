// MulberryAI - Frontend Logic
const API_URL = "http://localhost:5000";

const fileInput        = document.getElementById("fileInput");
const uploadZone       = document.getElementById("uploadZone");
const btnUpload        = document.getElementById("btnUpload");
const previewContainer = document.getElementById("previewContainer");
const previewImage     = document.getElementById("previewImage");
const clearBtn         = document.getElementById("clearBtn");
const analyzeBtn       = document.getElementById("analyzeBtn");
const btnText          = document.getElementById("btnText");
const btnLoader        = document.getElementById("btnLoader");
const resultsPlaceholder = document.getElementById("resultsPlaceholder");
const resultsContent     = document.getElementById("resultsContent");
const errorContent       = document.getElementById("errorContent");

let selectedFile = null;
let isDialogOpen = false;

const DISEASE_COLORS = {
  "Healthy":   "#22c55e",
  "Leaf Rust": "#f97316",
  "Leaf Spot": "#92400e"
};

// ===== File Dialog =====
function openFileDialog() {
  if (isDialogOpen) return;
  isDialogOpen = true;
  fileInput.click();
  setTimeout(() => { isDialogOpen = false; }, 1000);
}

btnUpload.addEventListener("click", (e) => { e.stopPropagation(); openFileDialog(); });
uploadZone.addEventListener("click", (e) => {
  if (e.target === btnUpload || btnUpload.contains(e.target)) return;
  openFileDialog();
});

fileInput.addEventListener("change", (e) => {
  isDialogOpen = false;
  const file = e.target.files[0];
  if (file) handleFile(file);
  fileInput.value = "";
});

// ===== Drag & Drop =====
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && (file.type.startsWith("image/") || file.name.toLowerCase().endsWith(".jfif")))
    handleFile(file);
});

// ===== Handle File =====
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    removeCanvas();
    uploadZone.style.display = "none";
    previewContainer.style.display = "block";
    analyzeBtn.disabled = false;
    resetResults();
  };
  reader.readAsDataURL(file);
}

// ===== Clear =====
clearBtn.addEventListener("click", resetUI);

function resetUI() {
  selectedFile = null;
  previewImage.src = "";
  removeCanvas();
  uploadZone.style.display = "block";
  previewContainer.style.display = "none";
  analyzeBtn.disabled = true;
  resetResults();
}

function resetResults() {
  resultsPlaceholder.style.display = "flex";
  resultsContent.style.display = "none";
  errorContent.style.display = "none";
  const cp = document.getElementById("confidencePanel");
  if (cp) cp.style.display = "none";
  const sp = document.getElementById("severityPanel");
  if (sp) sp.style.display = "none";
}

function removeCanvas() {
  const c = document.getElementById("annotationCanvas");
  if (c) c.remove();
}

// ===== Analyze =====
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setLoading(true);
  const formData = new FormData();
  formData.append("file", selectedFile);
  try {
    const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    displayResults(data);
    setTimeout(() => annotateImage(data), 100);
  } catch (err) {
    displayError(err.message);
  } finally {
    setLoading(false);
  }
});

// ===== Display Results =====
function displayResults(data) {
  resultsPlaceholder.style.display = "none";
  errorContent.style.display = "none";
  resultsContent.style.display = "block";

  const isHealthy = data.disease === "Healthy";
  const color = DISEASE_COLORS[data.disease] || "#f97316";

  if (isHealthy) {
    document.getElementById("resultIcon").innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="56" height="56"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>`;
  } else {
    document.getElementById("resultIcon").innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="56" height="56"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
  }

  // Status text — healthy or unhealthy
  const statusEl = document.getElementById("resultStatus");
  statusEl.textContent = isHealthy ? "Healthy Leaf" : "Unhealthy Leaf";
  statusEl.className = "result-status-text " + (isHealthy ? "status-healthy" : "status-diseased");

  // Disease name — only show if not healthy
  const diseaseEl = document.getElementById("diseaseName");
  if (isHealthy) {
    diseaseEl.textContent = "";
    diseaseEl.style.display = "none";
  } else {
    diseaseEl.textContent = `Disease Detected: ${data.disease}`;
    diseaseEl.style.display = "block";
    diseaseEl.style.color = color;
  }

  // Treatment
  document.getElementById("treatmentText") && (document.getElementById("treatmentText").textContent = data.treatment);

  // Disease detail section
  const detailEl  = document.getElementById("diseaseDetail");
  const healthyEl = document.getElementById("healthyMsg");

  if (isHealthy) {
    if (detailEl)  detailEl.style.display  = "none";
    if (healthyEl) healthyEl.style.display = "block";
  } else {
    if (healthyEl) healthyEl.style.display = "none";
    if (detailEl)  detailEl.style.display  = "block";
    const info = data.disease_info || {};
    document.getElementById("diseaseCause").textContent = info.cause || "";
    const whyEl = document.getElementById("diseaseWhy");
    whyEl.innerHTML = "";
    (info.why || []).forEach(item => {
      const li = document.createElement("li"); li.textContent = item; whyEl.appendChild(li);
    });
    const solEl = document.getElementById("diseaseSolution");
    solEl.innerHTML = "";
    (info.solution || []).forEach(item => {
      const li = document.createElement("li"); li.textContent = item; solEl.appendChild(li);
    });
  }

  // Confidence panel
  updateConfidencePanel(data);

  // Severity panel — only for unhealthy
  const severityPanel = document.getElementById("severityPanel");
  const severityHeader = document.getElementById("severityHeader");
  if (severityPanel && severityHeader) {
    if (!isHealthy) {
      severityHeader.textContent = "⚠ Immediate Action Required";
      severityPanel.style.display = "block";
    } else {
      severityPanel.style.display = "none";
    }
  }
}

// ===== Annotate Image =====
function annotateImage(data) {
  removeCanvas();
  const canvas = document.createElement("canvas");
  canvas.id = "annotationCanvas";
  canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;border-radius:12px;";
  canvas.width  = previewImage.offsetWidth;
  canvas.height = previewImage.offsetHeight;
  previewContainer.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const isHealthy = data.disease === "Healthy";
  const color = DISEASE_COLORS[data.disease] || "#f97316";

  if (isHealthy) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.shadowColor = color;
    ctx.shadowBlur = 20;
    ctx.strokeRect(6, 6, w - 12, h - 12);
    ctx.shadowBlur = 0;
    drawLabel(ctx, "✓ Healthy", w / 2 - 30, h - 20, color, true);
    return;
  }

  const seed = data.confidence * 137.5;
  const spots = generateSpots(w, h, seed, 3);

  spots.forEach((spot, i) => {
    ctx.beginPath();
    ctx.arc(spot.x, spot.y, spot.r + 8, 0, Math.PI * 2);
    ctx.strokeStyle = color + "44";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(spot.x, spot.y, spot.r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.beginPath();
    ctx.arc(spot.x, spot.y, spot.r, 0, Math.PI * 2);
    ctx.fillStyle = color + "25";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(spot.x, spot.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (i === 0) {
      const toRight = spot.x < w / 2;
      const lx = spot.x + (toRight ? spot.r + 50 : -(spot.r + 50));
      const ly = spot.y - 10;
      ctx.beginPath();
      ctx.moveTo(spot.x + (toRight ? spot.r : -spot.r), spot.y);
      ctx.lineTo(lx, ly);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      drawLabel(ctx, `⚠ ${data.disease}`, lx, ly - 6, color, toRight);
    }
  });

  // Disease stamp bottom right
  const stamp = data.disease;
  ctx.font = "bold 12px Inter,sans-serif";
  const sw = ctx.measureText(stamp).width + 16;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  roundRect(ctx, w - sw - 6, h - 30, sw, 22, 6);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textAlign = "right";
  ctx.fillText(stamp, w - 14, h - 14);
  ctx.textAlign = "left";
}

function generateSpots(w, h, seed, count) {
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const margin = 55;
  return Array.from({ length: count }, () => ({
    x: margin + rand() * (w - margin * 2),
    y: margin + rand() * (h - margin * 2),
    r: 18 + rand() * 20
  }));
}

function drawLabel(ctx, text, x, y, color, alignLeft = true) {
  ctx.font = "bold 12px Inter,sans-serif";
  const tw = ctx.measureText(text).width;
  const pad = 7;
  const bx = alignLeft ? x : x - tw - pad * 2;
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  roundRect(ctx, bx - pad, y - 16, tw + pad * 2, 22, 5);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  roundRect(ctx, bx - pad, y - 16, tw + pad * 2, 22, 5);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(text, bx, y);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ===== Confidence Panel =====
function updateConfidencePanel(data) {
  const panel = document.getElementById("confidencePanel");
  if (!panel) return;
  panel.style.display = "block";

  const conf = data.confidence || 0;
  document.getElementById("confValue").textContent = conf.toFixed(1) + "%";

  const fill = document.getElementById("confBarFill");
  fill.style.transition = "none";
  fill.style.width = "0%";
  fill.style.background = conf >= 85
    ? "linear-gradient(90deg,#4ade80,#16a34a)"
    : conf >= 65
      ? "linear-gradient(90deg,#fbbf24,#f97316)"
      : "linear-gradient(90deg,#f87171,#dc2626)";

  const breakdown = document.getElementById("confBreakdown");
  while (breakdown.firstChild) breakdown.removeChild(breakdown.firstChild);

  const preds = data.all_predictions || {};
  const sorted = Object.entries(preds).sort((a, b) => b[1] - a[1]);
  const bars = [];

  sorted.forEach(([label, pct]) => {
    const isTop = label === data.disease;

    const row = document.createElement("div");
    row.className = "conf-row";

    const labelEl = document.createElement("span");
    labelEl.className = "conf-row-label";
    labelEl.textContent = label;

    const track = document.createElement("div");
    track.className = "conf-row-track";

    const bar = document.createElement("div");
    bar.className = "conf-row-bar" + (isTop ? " top" : "");
    bar.style.width = "0%";
    bars.push({ bar, pct });

    const pctEl = document.createElement("span");
    pctEl.className = "conf-row-pct";
    pctEl.textContent = pct.toFixed(1) + "%";

    track.appendChild(bar);
    row.appendChild(labelEl);
    row.appendChild(track);
    row.appendChild(pctEl);
    breakdown.appendChild(row);
  });

  setTimeout(() => {
    fill.style.transition = "width 0.9s cubic-bezier(.4,0,.2,1)";
    fill.style.width = conf + "%";
    bars.forEach(({ bar, pct }) => {
      bar.style.transition = "width 0.7s cubic-bezier(.4,0,.2,1)";
      bar.style.width = pct + "%";
    });
  }, 80);
}


function displayError(message) {
  resultsPlaceholder.style.display = "none";
  resultsContent.style.display = "none";
  errorContent.style.display = "flex";
  document.getElementById("errorText").textContent =
    message.includes("Failed to fetch")
      ? "Cannot connect to backend. Make sure Flask is running on port 5000."
      : message;
}

// ===== Loading =====
function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.style.display = loading ? "none" : "inline";
  btnLoader.style.display = loading ? "inline-block" : "none";
}
