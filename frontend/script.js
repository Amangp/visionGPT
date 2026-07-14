const API_URL = "http://127.0.0.1:8000";
const form = document.getElementById("visionForm");
const imageInput = document.getElementById("imageInput");
const attachBtn = document.getElementById("attachBtn");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const previewCard = document.getElementById("previewCard");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeImageBtn = document.getElementById("removeImageBtn");
const hero = document.getElementById("hero");
const messages = document.getElementById("messages");
const chat = document.getElementById("chat");
const apiStatus = document.getElementById("apiStatus");
const history = document.getElementById("history");
const dropOverlay = document.getElementById("dropOverlay");
let selectedFile = null;

attachBtn.onclick = () => imageInput.click();
imageInput.onchange = () => setImage(imageInput.files[0]);
removeImageBtn.onclick = clearImage;
document.getElementById("newChatBtn").onclick = resetChat;
document.getElementById("menuBtn").onclick = () => document.getElementById("sidebar").classList.toggle("open");

function setImage(file) {
  if (!file || !["image/jpeg","image/png","image/webp"].includes(file.type)) return;
  selectedFile = file;
  imagePreview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
  previewCard.classList.remove("hidden");
  updateSend();
}
function clearImage() {
  selectedFile = null;
  imageInput.value = "";
  previewCard.classList.add("hidden");
  updateSend();
}
function updateSend(){
  sendBtn.disabled = !selectedFile && !questionInput.value.trim();
}
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight,150) + "px";
  updateSend();
});

["dragenter","dragover"].forEach(evt => window.addEventListener(evt, e => {
  e.preventDefault(); dropOverlay.classList.remove("hidden");
}));
["dragleave","drop"].forEach(evt => window.addEventListener(evt, e => {
  e.preventDefault(); dropOverlay.classList.add("hidden");
}));
window.addEventListener("drop", e => setImage(e.dataTransfer.files[0]));

form.addEventListener("submit", async e => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!selectedFile && !question) return;
  const imageURL = selectedFile ? URL.createObjectURL(selectedFile) : null;
  hero.classList.add("hidden");
  addUserMessage(question || "Describe this image.", imageURL);
  addHistory(question || "Image description");
  const thinking = addThinking();

  const data = new FormData();
  if (selectedFile) data.append("image", selectedFile);
  if (question) data.append("question", question);

  sendBtn.disabled = true;
  try {
    const response = await fetch(`${API_URL}/api/vision`, { method:"POST", body:data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "VisionGPT request failed");
    thinking.remove();
    addAssistantMessage(payload.answer);
  } catch (error) {
    thinking.remove();
    addAssistantMessage(`Connection error: ${error.message}. Make sure the FastAPI server is running.`);
  } finally {
    clearImage();
    questionInput.value = "";
    questionInput.style.height = "auto";
  }
});

function addUserMessage(text, imageURL){
  const el = document.createElement("div");
  el.className = "message user";
  const imageHTML = imageURL ? `<img class="message-image" src="${imageURL}" alt="Uploaded image">` : "";
  el.innerHTML = `<div class="avatar">You</div><div class="bubble">${imageHTML}<div>${escapeHTML(text)}</div></div>`;
  messages.appendChild(el); scrollBottom();
}
function addAssistantMessage(text){
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `<div class="avatar">V</div><div class="bubble">${escapeHTML(text)}</div>`;
  messages.appendChild(el); scrollBottom();
}
function addThinking(){
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `<div class="avatar">V</div><div class="bubble"><div class="thinking"><i></i><i></i><i></i></div></div>`;
  messages.appendChild(el); scrollBottom(); return el;
}
function addHistory(text){
  const el = document.createElement("div");
  el.className = "history-item";
  el.textContent = text;
  history.prepend(el);
}
function resetChat(){ messages.innerHTML=""; history.innerHTML=""; hero.classList.remove("hidden"); clearImage(); questionInput.value=""; }
function scrollBottom(){ requestAnimationFrame(() => chat.scrollTo({top:chat.scrollHeight,behavior:"smooth"})); }
function escapeHTML(value){ const d=document.createElement("div"); d.textContent=value; return d.innerHTML; }

async function checkAPI(){
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) throw new Error();
    apiStatus.className="api-status online";
    apiStatus.querySelector("span").textContent="API online";
  } catch {
    apiStatus.className="api-status offline";
    apiStatus.querySelector("span").textContent="API offline";
  }
}
checkAPI();
