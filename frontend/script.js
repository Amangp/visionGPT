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

const sidebar = document.getElementById("sidebar");
const menuBtn = document.getElementById("menuBtn");
const newChatBtn = document.getElementById("newChatBtn");


const CHAT_STORAGE_KEY = "visiongpt_chats";


let selectedFile = null;
let selectedImageId = null;
let sessionId = null;

let currentChatId = null;

let chats = loadChats();


// ==========================================
// CHAT STORAGE
// ==========================================

function loadChats() {

  try {

    const storedChats = localStorage.getItem(
      CHAT_STORAGE_KEY
    );


    if (!storedChats) {

      return {};

    }


    return JSON.parse(
      storedChats
    );

  } catch (error) {

    console.error(
      "Failed to load chats:",
      error
    );


    return {};

  }

}


function saveChats() {

  localStorage.setItem(

    CHAT_STORAGE_KEY,

    JSON.stringify(chats)

  );

}


// ==========================================
// CREATE CHAT
// ==========================================

function createChat(title) {

  const chatId = crypto.randomUUID();


  chats[chatId] = {

    id: chatId,

    title: title || "New analysis",

    createdAt: Date.now(),

    updatedAt: Date.now(),

    messages: []

  };


  currentChatId = chatId;


  saveChats();

  renderHistory();


  return chats[chatId];

}


// ==========================================
// GET CURRENT CHAT
// ==========================================

function getCurrentChat() {

  if (
    !currentChatId ||
    !chats[currentChatId]
  ) {

    return null;

  }


  return chats[currentChatId];

}


// ==========================================
// SAVE MESSAGE
// ==========================================

function saveMessage(
  role,
  text,
  imageId = null
) {

  let currentChat = getCurrentChat();


  if (!currentChat) {

    currentChat = createChat(
      text || "Image analysis"
    );

  }


  currentChat.messages.push({

    role,

    text,

    imageId,

    createdAt: Date.now()

  });


  currentChat.updatedAt = Date.now();


  saveChats();

  renderHistory();

}


// ==========================================
// RENDER HISTORY
// ==========================================

function renderHistory() {

  history.innerHTML = "";


  const sortedChats = Object.values(chats)

    .sort(

      (firstChat, secondChat) =>

        secondChat.updatedAt -
        firstChat.updatedAt

    );


  sortedChats.forEach(
    (savedChat) => {

      const element = document.createElement(
        "div"
      );


      element.className = "history-item";


      if (
        savedChat.id === currentChatId
      ) {

        element.classList.add(
          "active"
        );

      }


      element.textContent = (
        savedChat.title
      );


      element.onclick = () => {

        openChat(
          savedChat.id
        );

      };


      history.appendChild(
        element
      );

    }
  );

}


// ==========================================
// OPEN CHAT
// ==========================================

async function openChat(chatId) {

  const savedChat = chats[chatId];


  if (!savedChat) {

    return;

  }


  currentChatId = chatId;

  sessionId = null;


  clearImage();


  questionInput.value = "";

  questionInput.style.height = "auto";


  messages.innerHTML = "";


  hero.classList.add(
    "hidden"
  );


  for (
    const message of savedChat.messages
  ) {

    if (
      message.role === "user"
    ) {

      let restoredImageURL = null;


      if (message.imageId) {

        try {

          const imageRecord = await getChatImage(
            message.imageId
          );


          if (
            imageRecord &&
            imageRecord.blob
          ) {

            restoredImageURL = URL.createObjectURL(
              imageRecord.blob
            );

          }

        } catch (error) {

          console.error(
            "Image restore failed:",
            error
          );

        }

      }


      addUserMessage(
        message.text,
        restoredImageURL,
        false,
        message.imageId
      );


    } else {


      addAssistantMessage(
        message.text,
        false
      );

    }

  }


  renderHistory();

  updateSend();

  scrollBottom();

}


// ==========================================
// IMAGE UPLOAD
// ==========================================

attachBtn.onclick = () => {

  imageInput.click();

};


imageInput.onchange = () => {

  setImage(
    imageInput.files[0]
  );

};


removeImageBtn.onclick = () => {

  clearImage();

};


// ==========================================
// SIDEBAR
// ==========================================

newChatBtn.onclick = () => {

  resetChat();

};


menuBtn.onclick = () => {

  sidebar.classList.toggle(
    "open"
  );

};


// ==========================================
// IMAGE HANDLING
// ==========================================

async function setImage(file) {

  if (!file) {

    return;

  }


  const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/webp"
  ];


  if (
    !allowedTypes.includes(file.type)
  ) {

    addAssistantMessage(
      "Please upload a JPG, PNG, or WEBP image."
    );

    return;

  }


  selectedFile = file;


  selectedImageId = await saveChatImage(
    file
  );


  imagePreview.src = URL.createObjectURL(
    file
  );


  fileName.textContent = file.name;


  fileSize.textContent = (
    `${(file.size / 1024).toFixed(1)} KB`
  );


  previewCard.classList.remove(
    "hidden"
  );


  updateSend();

}


// ==========================================
// CLEAR IMAGE
// ==========================================

function clearImage() {

  selectedFile = null;

  selectedImageId = null;

  imageInput.value = "";

  previewCard.classList.add(
    "hidden"
  );

  updateSend();

}

// ==========================================
// SEND BUTTON STATE
// ==========================================

function updateSend() {

  const hasImage = Boolean(
    selectedFile
  );


  const hasQuestion = Boolean(
    questionInput.value.trim()
  );


  sendBtn.disabled = (

    !hasImage &&
    !hasQuestion

  );

}


// ==========================================
// QUESTION INPUT
// ==========================================

// ==========================================
// ENTER TO SEND
// ==========================================

questionInput.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();


      if (!sendBtn.disabled) {

        form.requestSubmit();

      }

    }

  }
);


// ==========================================
// DRAG AND DROP
// ==========================================

[
  "dragenter",
  "dragover"
].forEach(

  (eventName) => {

    window.addEventListener(

      eventName,

      (event) => {

        event.preventDefault();


        dropOverlay.classList.remove(
          "hidden"
        );

      }

    );

  }

);


[
  "dragleave",
  "drop"
].forEach(

  (eventName) => {

    window.addEventListener(

      eventName,

      (event) => {

        event.preventDefault();


        dropOverlay.classList.add(
          "hidden"
        );

      }

    );

  }

);


window.addEventListener(

  "drop",

  (event) => {

    const file = (

      event.dataTransfer.files[0]

    );


    setImage(file);

  }

);


// ==========================================
// FORM SUBMIT
// ==========================================

form.addEventListener(
  "submit",
  async (event) => {

    event.preventDefault();

    const question = questionInput.value.trim();

    const currentFile = selectedFile;

    const currentImageId = selectedImageId;

    if (
      !currentFile &&
      !question
    ) {
      return;
    }

    const imageURL = currentFile
      ? URL.createObjectURL(currentFile)
      : null;

    const userText =
      question ||
      "Describe this image.";

    if (!currentChatId) {
      createChat(userText);
    }

    hero.classList.add(
      "hidden"
    );

    addUserMessage(
      userText,
      imageURL,
      true,
      currentImageId
    );

    const thinking = addThinking();

    const data = new FormData();

    if (currentFile) {
      data.append(
        "image",
        currentFile
      );
    }

    if (question) {
      data.append(
        "question",
        question
      );
    }

    if (sessionId) {
      data.append(
        "session_id",
        sessionId
      );
    }

    console.log(
      "Sending session:",
      sessionId
    );

    sendBtn.disabled = true;

    try {

      const response = await fetch(
        `${API_URL}/api/vision`,
        {
          method: "POST",
          body: data
        }
      );

      const payload =
        await response.json();

      console.log(
        "VisionGPT response:",
        payload
      );

      if (!response.ok) {
        throw new Error(
          payload.error?.message ||
          payload.detail ||
          "VisionGPT request failed"
        );
      }

      if (payload.session_id) {
        sessionId =
          payload.session_id;

        console.log(
          "VisionGPT session saved:",
          sessionId
        );
      }

      thinking.remove();

      addAssistantMessage(
        payload.answer
      );

    } catch (error) {

      thinking.remove();

      console.error(
        "VisionGPT error:",
        error
      );

      addAssistantMessage(
        `Connection error: ${error.message}`
      );

    } finally {

      clearImage();

      questionInput.value = "";

      questionInput.style.height =
        "auto";

      updateSend();

    }

  }
);

// ==========================================
// USER MESSAGE
// ==========================================

function addUserMessage(
  text,
  imageURL,
  persist = true,
  imageId = null
) {

  const element = document.createElement(
    "div"
  );


  element.className = "message user";


  const imageHTML = imageURL

    ? `

      <img
        class="message-image"
        src="${imageURL}"
        alt="Uploaded image"
      >

    `

    : "";


  element.innerHTML = `

    <div class="avatar">
      You
    </div>

    <div class="bubble">

      ${imageHTML}

      <div>
        ${escapeHTML(text)}
      </div>

    </div>

  `;


  messages.appendChild(
    element
  );


  if (persist) {

    saveMessage(
      "user",
      text,
      imageId
    );

  }


  scrollBottom();

}


// ==========================================
// ASSISTANT MESSAGE
// ==========================================

function addAssistantMessage(

  text,

  persist = true

) {

  const element = (

    document.createElement("div")

  );


  element.className = (
    "message assistant"
  );


  element.innerHTML = `

    <div class="avatar">
      V
    </div>

    <div class="bubble">

      ${escapeHTML(text)}

    </div>

  `;


  messages.appendChild(
    element
  );


  if (persist) {

    saveMessage(

      "assistant",

      text

    );

  }


  scrollBottom();

}


// ==========================================
// THINKING
// ==========================================

function addThinking() {

  const element = (

    document.createElement("div")

  );


  element.className = (
    "message assistant"
  );


  element.innerHTML = `

    <div class="avatar">
      V
    </div>

    <div class="bubble">

      <div class="thinking">

        <i></i>

        <i></i>

        <i></i>

      </div>

    </div>

  `;


  messages.appendChild(
    element
  );


  scrollBottom();


  return element;

}


// ==========================================
// NEW ANALYSIS
// ==========================================

async function resetChat() {

  const activeSessionId = sessionId;


  sessionId = null;

  currentChatId = null;


  if (activeSessionId) {

    try {

      await fetch(

        `${API_URL}/api/sessions/${activeSessionId}`,

        {

          method: "DELETE"

        }

      );

    } catch (error) {

      console.error(

        "Session cleanup failed:",

        error

      );

    }

  }


  messages.innerHTML = "";


  hero.classList.remove(
    "hidden"
  );


  clearImage();


  questionInput.value = "";


  questionInput.style.height = "auto";


  updateSend();

  renderHistory();

}


// ==========================================
// SCROLL
// ==========================================

function scrollBottom() {

  requestAnimationFrame(

    () => {

      chat.scrollTo({

        top: chat.scrollHeight,

        behavior: "smooth"

      });

    }

  );

}


// ==========================================
// HTML ESCAPE
// ==========================================

function escapeHTML(value) {

  const div = (

    document.createElement("div")

  );


  div.textContent = value;


  return div.innerHTML;

}


// ==========================================
// API HEALTH
// ==========================================

async function checkAPI() {

  try {

    const response = await fetch(

      `${API_URL}/health`

    );


    if (!response.ok) {

      throw new Error(
        "API unavailable"
      );

    }


    apiStatus.className = (
      "api-status online"
    );


    apiStatus
      .querySelector("span")
      .textContent = "API online";


  } catch (error) {

    console.error(

      "API health error:",

      error

    );


    apiStatus.className = (
      "api-status offline"
    );


    apiStatus
      .querySelector("span")
      .textContent = "API offline";

  }

}


// ==========================================
// STARTUP
// ==========================================

renderHistory();

checkAPI();

updateSend();