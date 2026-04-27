// ============================================
// SMART SPAM DETECTOR — BACKGROUND SERVICE WORKER
// ============================================

const DEFAULT_API_URL = "http://localhost:5000";

// Get the API URL from storage
async function getApiUrl() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["apiUrl"], (result) => {
      resolve(result.apiUrl || DEFAULT_API_URL);
    });
  });
}

// Handle messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYZE_EMAIL") {
    analyzeEmail(message.text, message.gmail_spam, message.sender)
      .then((result) => sendResponse({ success: true, data: result }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }
});

// Call the prediction API
async function analyzeEmail(text, gmailSpam, senderInfo) {
  const apiUrl = await getApiUrl();

  const response = await fetch(`${apiUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      gmail_spam: gmailSpam || false,
      sender: senderInfo || ""
    }),
  });

  if (!response.ok) {
    throw new Error(`API returned ${response.status}`);
  }

  return response.json();
}
