// ============================================
// SMART SPAM DETECTOR — POPUP SCRIPT
// ============================================

const DEFAULT_API_URL = "http://localhost:5000";

// ---- DOM ELEMENTS ----
const elements = {
  emailText: document.getElementById("emailText"),
  charCount: document.getElementById("charCount"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  btnLoader: document.getElementById("btnLoader"),
  resultCard: document.getElementById("resultCard"),
  resultHeader: document.getElementById("resultHeader"),
  resultIcon: document.getElementById("resultIcon"),
  resultLabel: document.getElementById("resultLabel"),
  confidenceBar: document.getElementById("confidenceBar"),
  confidenceText: document.getElementById("confidenceText"),
  triggerWords: document.getElementById("triggerWords"),
  triggerTags: document.getElementById("triggerTags"),
  statusBadge: document.getElementById("statusBadge"),
  apiUrl: document.getElementById("apiUrl"),
  saveSettings: document.getElementById("saveSettings"),
  saveToast: document.getElementById("saveToast"),
  clearHistory: document.getElementById("clearHistory"),
  historyList: document.getElementById("historyList"),
};

// ---- TAB NAVIGATION ----
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---- CHARACTER COUNT ----
elements.emailText.addEventListener("input", () => {
  elements.charCount.textContent = elements.emailText.value.length;
});

// ---- GET API URL ----
async function getApiUrl() {
  return new Promise((resolve) => {
    if (typeof chrome !== "undefined" && chrome.storage) {
      chrome.storage.local.get(["apiUrl"], (result) => {
        resolve(result.apiUrl || DEFAULT_API_URL);
      });
    } else {
      resolve(DEFAULT_API_URL);
    }
  });
}

// ---- CHECK API STATUS ----
async function checkApiStatus() {
  try {
    const apiUrl = await getApiUrl();
    const response = await fetch(apiUrl, { method: "GET", signal: AbortSignal.timeout(5000) });
    const data = await response.json();
    if (data.status === "ok") {
      elements.statusBadge.classList.remove("offline");
      elements.statusBadge.innerHTML = '<span class="status-dot"></span><span>Online</span>';
    } else {
      setOffline();
    }
  } catch {
    setOffline();
  }
}

function setOffline() {
  elements.statusBadge.classList.add("offline");
  elements.statusBadge.innerHTML = '<span class="status-dot"></span><span>Offline</span>';
}

// ---- ANALYZE ----
elements.analyzeBtn.addEventListener("click", async () => {
  const text = elements.emailText.value.trim();

  if (!text) {
    shakeButton();
    return;
  }

  // Start loading
  elements.analyzeBtn.classList.add("loading");
  elements.resultCard.classList.add("hidden");

  try {
    const apiUrl = await getApiUrl();
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    showResult(data);
    saveToHistory(text, data);
  } catch (err) {
    showError(err.message);
  } finally {
    elements.analyzeBtn.classList.remove("loading");
  }
});

// ---- SHOW RESULT ----
function showResult(data) {
  const isSpam = data.prediction === "spam";

  elements.resultCard.classList.remove("hidden");
  elements.resultHeader.className = `result-header ${isSpam ? "spam" : "genuine"}`;
  elements.resultIcon.textContent = isSpam ? "🚨" : "✅";
  elements.resultLabel.textContent = isSpam ? "Spam Detected" : "Genuine Email";

  // Confidence bar
  elements.confidenceBar.className = `confidence-bar ${isSpam ? "spam" : "genuine"}`;
  setTimeout(() => {
    elements.confidenceBar.style.width = `${data.confidence}%`;
  }, 100);
  elements.confidenceText.textContent = `${data.confidence}%`;
  elements.confidenceText.style.color = isSpam ? "var(--accent-red)" : "var(--accent-green)";

  // Trigger words
  if (data.trigger_words && data.trigger_words.length > 0) {
    elements.triggerWords.classList.remove("hidden");
    elements.triggerTags.innerHTML = data.trigger_words
      .map((w) => `<span class="trigger-tag">${w}</span>`)
      .join("");
  } else {
    elements.triggerWords.classList.add("hidden");
  }
}

// ---- SHOW ERROR ----
function showError(message) {
  elements.resultCard.classList.remove("hidden");
  elements.resultHeader.className = "result-header spam";
  elements.resultIcon.textContent = "⚠️";
  elements.resultLabel.textContent = "Connection Error";
  elements.confidenceBar.style.width = "0%";
  elements.confidenceText.textContent = message;
  elements.confidenceText.style.color = "var(--accent-red)";
  elements.triggerWords.classList.add("hidden");
}

// ---- SHAKE ANIMATION ----
function shakeButton() {
  elements.analyzeBtn.style.animation = "shake 0.4s ease";
  setTimeout(() => {
    elements.analyzeBtn.style.animation = "";
  }, 400);
}

// ---- HISTORY ----
async function saveToHistory(text, data) {
  const history = await getHistory();
  history.unshift({
    text: text.substring(0, 100),
    prediction: data.prediction,
    confidence: data.confidence,
    timestamp: new Date().toLocaleString(),
  });

  // Keep last 20
  if (history.length > 20) history.pop();

  if (typeof chrome !== "undefined" && chrome.storage) {
    chrome.storage.local.set({ scanHistory: history });
  }
  renderHistory(history);
}

async function getHistory() {
  return new Promise((resolve) => {
    if (typeof chrome !== "undefined" && chrome.storage) {
      chrome.storage.local.get(["scanHistory"], (result) => {
        resolve(result.scanHistory || []);
      });
    } else {
      resolve([]);
    }
  });
}

function renderHistory(history) {
  if (!history.length) {
    elements.historyList.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">📭</span>
        <p>No scans yet. Analyze an email to see results here.</p>
      </div>`;
    return;
  }

  elements.historyList.innerHTML = history
    .map(
      (item) => `
    <div class="history-item">
      <div class="history-badge ${item.prediction}">
        ${item.prediction === "spam" ? "🚨" : "✅"}
      </div>
      <div class="history-info">
        <div class="history-text">${escapeHtml(item.text)}</div>
        <div class="history-meta">${item.prediction.toUpperCase()} • ${item.confidence}% • ${item.timestamp}</div>
      </div>
    </div>`
    )
    .join("");
}

elements.clearHistory.addEventListener("click", () => {
  if (typeof chrome !== "undefined" && chrome.storage) {
    chrome.storage.local.set({ scanHistory: [] });
  }
  renderHistory([]);
});

// ---- SETTINGS ----
elements.saveSettings.addEventListener("click", async () => {
  const url = elements.apiUrl.value.trim().replace(/\/+$/, ""); // remove trailing slash
  if (typeof chrome !== "undefined" && chrome.storage) {
    chrome.storage.local.set({ apiUrl: url || DEFAULT_API_URL });
  }
  elements.saveToast.classList.remove("hidden");
  setTimeout(() => elements.saveToast.classList.add("hidden"), 2500);

  // Re-check API status
  checkApiStatus();
});

// ---- LOAD SETTINGS ----
async function loadSettings() {
  const url = await getApiUrl();
  elements.apiUrl.value = url;
}

// ---- ESCAPE HTML ----
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---- LISTEN FOR CONTENT SCRIPT MESSAGES ----
if (typeof chrome !== "undefined" && chrome.runtime) {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "EMAIL_SCANNED") {
      // Switch to scan tab and show result
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      document.querySelector('[data-tab="scan"]').classList.add("active");
      document.getElementById("tab-scan").classList.add("active");

      elements.emailText.value = message.text.substring(0, 5000);
      elements.charCount.textContent = elements.emailText.value.length;
      showResult(message.result);
    }
  });
}

// ---- INIT ----
(async () => {
  await loadSettings();
  checkApiStatus();
  const history = await getHistory();
  renderHistory(history);
})();
