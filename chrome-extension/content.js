// ============================================
// SMART SPAM DETECTOR — GMAIL CONTENT SCRIPT
// Floating "Check Spam" button on Gmail emails
// ============================================

(function () {
  "use strict";

  const BUTTON_ID = "ssd-floating-btn";
  const RESULT_ID = "ssd-result-overlay";
  let isAnalyzing = false;
  let lastUrl = "";

  // ---- MAIN LOOP: Poll for email view changes ----
  // Gmail is a SPA — URL changes when opening emails but page doesn't reload
  setInterval(checkForEmailView, 1000);

  function checkForEmailView() {
    const currentUrl = window.location.href;

    // Gmail email view URLs contain a message ID hash fragment
    // Inbox: https://mail.google.com/mail/u/0/#inbox
    // Email: https://mail.google.com/mail/u/0/#inbox/FMfcgzQXKJWPsNqfnqdnVHLjTZBRhBjD
    const isEmailOpen = isEmailViewOpen();

    const btn = document.getElementById(BUTTON_ID);

    if (isEmailOpen && !btn) {
      injectFloatingButton();
    } else if (!isEmailOpen && btn) {
      btn.remove();
      const result = document.getElementById(RESULT_ID);
      if (result) result.remove();
    }

    lastUrl = currentUrl;
  }

  // ---- DETECT IF AN EMAIL IS OPEN ----
  function isEmailViewOpen() {
    // Method 1: Check URL pattern — email URLs have a longer hash
    const hash = window.location.hash;
    // Email view hashes look like #inbox/MESSAGE_ID or #sent/MESSAGE_ID etc.
    // They have at least 2 path segments after #
    const hashParts = hash.replace("#", "").split("/");
    if (hashParts.length >= 2 && hashParts[1].length > 5) {
      return true;
    }

    // Method 2: Check for email body elements in DOM
    if (document.querySelector('div.a3s')) return true;
    if (document.querySelector('[data-message-id]')) return true;
    if (document.querySelector('div[role="list"] div[role="listitem"]')) return true;

    // Method 3: Check for conversation view indicators
    if (document.querySelector('table.Bs > tr > td > div[class*="a3s"]')) return true;
    if (document.querySelector('.ii.gt')) return true;
    if (document.querySelector('.nH.if')) return true;

    return false;
  }

  // ---- INJECT FLOATING BUTTON ----
  function injectFloatingButton() {
    // Don't duplicate
    if (document.getElementById(BUTTON_ID)) return;

    const btn = document.createElement("div");
    btn.id = BUTTON_ID;
    btn.innerHTML = `
      <div class="ssd-fab-inner">
        <span class="ssd-fab-shield">🛡️</span>
        <span class="ssd-fab-label">Check Spam</span>
      </div>
    `;
    btn.addEventListener("click", handleCheckSpam);
    document.body.appendChild(btn);
  }

  // ---- HANDLE CHECK SPAM ----
  async function handleCheckSpam() {
    if (isAnalyzing) return;
    isAnalyzing = true;

    const btn = document.getElementById(BUTTON_ID);
    if (btn) {
      btn.classList.add("ssd-loading");
      btn.querySelector(".ssd-fab-label").textContent = "Scanning...";
    }

    // Remove previous result
    const prevResult = document.getElementById(RESULT_ID);
    if (prevResult) prevResult.remove();

    try {
      const emailText = extractEmailText();

      if (!emailText || emailText.length < 5) {
        showResultOverlay({
          prediction: "error",
          confidence: 0,
          trigger_words: [],
        }, "Could not extract email text. Try the popup instead.");
        return;
      }

      // Detect Gmail context clues
      const gmailSpamLabel = isInSpamFolder();
      const senderInfo = extractSender();

      // Send to background script → API
      const response = await chrome.runtime.sendMessage({
        type: "ANALYZE_EMAIL",
        text: emailText,
        gmail_spam: gmailSpamLabel,
        sender: senderInfo,
      });

      if (response && response.success) {
        showResultOverlay(response.data);
        // Also save to history
        saveToHistory(emailText, response.data);
      } else {
        showResultOverlay(
          { prediction: "error", confidence: 0, trigger_words: [] },
          (response && response.error) || "API connection failed. Check Settings."
        );
      }
    } catch (err) {
      showResultOverlay(
        { prediction: "error", confidence: 0, trigger_words: [] },
        "Extension error: " + err.message
      );
    } finally {
      isAnalyzing = false;
      if (btn) {
        btn.classList.remove("ssd-loading");
        btn.querySelector(".ssd-fab-label").textContent = "Check Spam";
      }
    }
  }

  // ---- DETECT GMAIL SPAM FOLDER ----
  function isInSpamFolder() {
    // Check URL
    const hash = window.location.hash.toLowerCase();
    if (hash.includes("spam")) return true;

    // Check for spam label in the email header
    const labels = document.querySelectorAll('.ar .at, .aY2 .at, .at');
    for (const label of labels) {
      const text = (label.textContent || "").toLowerCase().trim();
      if (text === "spam" || text === "junk") return true;
    }

    // Check for Gmail's spam warning banner
    const warningText = document.querySelector('.bAs, .gE');
    if (warningText) {
      const wt = (warningText.textContent || "").toLowerCase();
      if (wt.includes("spam") || wt.includes("phishing") || wt.includes("suspicious")) return true;
    }

    return false;
  }

  // ---- EXTRACT SENDER INFO ----
  function extractSender() {
    // Try to get sender email from header
    const senderEl = document.querySelector('.gD[email], [email]');
    if (senderEl) return senderEl.getAttribute("email") || "";

    // Try from "from" display
    const fromEl = document.querySelector('.go, .gD');
    if (fromEl) return (fromEl.textContent || "").trim();

    return "";
  }

  // ---- EXTRACT EMAIL TEXT ----
  function extractEmailText() {
    let parts = [];

    // Grab the email subject (strong spam signal)
    const subjectEl = document.querySelector('h2[data-thread-perm-id]') ||
                      document.querySelector('div[role="main"] h2') ||
                      document.querySelector('.hP');
    if (subjectEl) {
      const subject = (subjectEl.innerText || "").trim();
      if (subject) parts.push("Subject: " + subject);
    }

    // Gmail spam/phishing warning banner
    const warningBanner = document.querySelector('.bAs');
    if (warningBanner) {
      parts.push(warningBanner.innerText || "");
    }

    // Strategy 1: Gmail message body (.a3s.aiL or .a3s) — most reliable
    const bodies = document.querySelectorAll("div.a3s.aiL, div.a3s");
    if (bodies.length > 0) {
      const bodyText = Array.from(bodies)
        .map((el) => el.innerText || el.textContent)
        .join("\n\n")
        .trim();
      if (bodyText.length > 5) parts.push(bodyText);
    }

    // Strategy 2: .ii.gt class (Gmail message body wrapper)
    if (parts.length <= 1) {
      const gtBodies = document.querySelectorAll(".ii.gt");
      if (gtBodies.length > 0) {
        const gtText = Array.from(gtBodies)
          .map((el) => el.innerText || el.textContent)
          .join("\n\n")
          .trim();
        if (gtText.length > 5) parts.push(gtText);
      }
    }

    // Strategy 3: data-message-id containers
    if (parts.length <= 1) {
      const msgContainers = document.querySelectorAll("[data-message-id]");
      if (msgContainers.length > 0) {
        const msgText = Array.from(msgContainers)
          .map((el) => el.innerText || el.textContent)
          .join("\n\n")
          .trim();
        if (msgText.length > 5) parts.push(msgText);
      }
    }

    // Strategy 4: Fallback — main content area
    if (parts.length <= 1) {
      const mainArea = document.querySelector('div[role="main"]');
      if (mainArea) {
        const mainText = (mainArea.innerText || "").trim();
        if (mainText.length > 10) parts.push(mainText);
      }
    }

    let fullText = parts.join("\n\n").trim();

    // Strip common Gmail UI noise
    const uiNoise = [
      "Reply", "Forward", "Report spam", "Report not spam", "Delete forever",
      "Not spam", "Compose", "Inbox", "Starred", "Snoozed", "Sent", "Drafts",
      "More", "Labels", "Manage labels", "Create new label", "Show details",
      "Hide details", "to me", "Unsubscribe"
    ];
    for (const noise of uiNoise) {
      fullText = fullText.replace(new RegExp("^" + noise + "$", "gm"), "");
    }

    // Clean up extra whitespace
    fullText = fullText.replace(/\n{3,}/g, "\n\n").trim();

    return fullText.substring(0, 5000) || null;
  }

  // ---- SHOW RESULT OVERLAY ----
  function showResultOverlay(data, errorMsg) {
    const existing = document.getElementById(RESULT_ID);
    if (existing) existing.remove();

    const isSpam = data.prediction === "spam";
    const isError = data.prediction === "error";

    const overlay = document.createElement("div");
    overlay.id = RESULT_ID;
    overlay.className = `ssd-overlay ${isError ? "ssd-error" : isSpam ? "ssd-spam" : "ssd-genuine"}`;

    let triggerHTML = "";
    if (data.trigger_words && data.trigger_words.length > 0) {
      triggerHTML = `
        <div class="ssd-overlay-triggers">
          <span class="ssd-trigger-title">⚠️ Trigger words:</span>
          <div class="ssd-trigger-list">
            ${data.trigger_words.map((w) => `<span class="ssd-tag">${w}</span>`).join("")}
          </div>
        </div>`;
    }

    overlay.innerHTML = `
      <button class="ssd-overlay-close" title="Close">&times;</button>
      <div class="ssd-overlay-header">
        <span class="ssd-overlay-icon">${isError ? "⚠️" : isSpam ? "🚨" : "✅"}</span>
        <div>
          <div class="ssd-overlay-title">${isError ? "Error" : isSpam ? "Spam Detected!" : "Genuine Email"}</div>
          <div class="ssd-overlay-sub">${isError ? (errorMsg || "Unknown error") : `Confidence: ${data.confidence}%`}</div>
        </div>
      </div>
      ${!isError ? `
      <div class="ssd-overlay-bar-wrap">
        <div class="ssd-overlay-bar ${isSpam ? "spam" : "genuine"}" style="width: 0%"></div>
      </div>` : ""}
      ${triggerHTML}
    `;

    document.body.appendChild(overlay);

    // Animate bar
    if (!isError) {
      requestAnimationFrame(() => {
        setTimeout(() => {
          const bar = overlay.querySelector(".ssd-overlay-bar");
          if (bar) bar.style.width = `${data.confidence}%`;
        }, 50);
      });
    }

    // Close button
    overlay.querySelector(".ssd-overlay-close").addEventListener("click", () => {
      overlay.classList.add("ssd-closing");
      setTimeout(() => overlay.remove(), 300);
    });

    // Auto dismiss after 20s
    setTimeout(() => {
      const el = document.getElementById(RESULT_ID);
      if (el) {
        el.classList.add("ssd-closing");
        setTimeout(() => el.remove(), 300);
      }
    }, 20000);
  }

  // ---- SAVE TO HISTORY ----
  function saveToHistory(text, data) {
    chrome.storage.local.get(["scanHistory"], (result) => {
      const history = result.scanHistory || [];
      history.unshift({
        text: text.substring(0, 100),
        prediction: data.prediction,
        confidence: data.confidence,
        timestamp: new Date().toLocaleString(),
      });
      if (history.length > 20) history.pop();
      chrome.storage.local.set({ scanHistory: history });
    });
  }
})();
