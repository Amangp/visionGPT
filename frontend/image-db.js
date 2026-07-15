const VISIONGPT_DB_NAME = "VisionGPTDB";
const VISIONGPT_DB_VERSION = 1;
const IMAGE_STORE_NAME = "chatImages";


function openVisionGPTDB() {

  return new Promise(
    (resolve, reject) => {

      const request = indexedDB.open(
        VISIONGPT_DB_NAME,
        VISIONGPT_DB_VERSION
      );


      request.onupgradeneeded = (event) => {

        const database = event.target.result;


        if (
          !database.objectStoreNames.contains(
            IMAGE_STORE_NAME
          )
        ) {

          database.createObjectStore(
            IMAGE_STORE_NAME,
            {
              keyPath: "imageId"
            }
          );

        }

      };


      request.onsuccess = () => {

        resolve(request.result);

      };


      request.onerror = () => {

        reject(request.error);

      };

    }
  );

}


async function saveChatImage(file) {

  const database = await openVisionGPTDB();

  const imageId = crypto.randomUUID();


  return new Promise(
    (resolve, reject) => {

      const transaction = database.transaction(
        IMAGE_STORE_NAME,
        "readwrite"
      );


      const store = transaction.objectStore(
        IMAGE_STORE_NAME
      );


      store.put({

        imageId,

        blob: file,

        name: file.name,

        type: file.type,

        createdAt: Date.now()

      });


      transaction.oncomplete = () => {

        database.close();

        resolve(imageId);

      };


      transaction.onerror = () => {

        database.close();

        reject(transaction.error);

      };

    }
  );

}


async function getChatImage(imageId) {

  if (!imageId) {

    return null;

  }


  const database = await openVisionGPTDB();


  return new Promise(
    (resolve, reject) => {

      const transaction = database.transaction(
        IMAGE_STORE_NAME,
        "readonly"
      );


      const store = transaction.objectStore(
        IMAGE_STORE_NAME
      );


      const request = store.get(
        imageId
      );


      request.onsuccess = () => {

        database.close();

        resolve(
          request.result || null
        );

      };


      request.onerror = () => {

        database.close();

        reject(request.error);

      };

    }
  );

}


async function deleteChatImage(imageId) {

  if (!imageId) {

    return;

  }


  const database = await openVisionGPTDB();


  return new Promise(
    (resolve, reject) => {

      const transaction = database.transaction(
        IMAGE_STORE_NAME,
        "readwrite"
      );


      transaction
        .objectStore(IMAGE_STORE_NAME)
        .delete(imageId);


      transaction.oncomplete = () => {

        database.close();

        resolve();

      };


      transaction.onerror = () => {

        database.close();

        reject(transaction.error);

      };

    }
  );

}