<script setup>
import { state as socketState, socket } from "@/socket";
import { computed } from "vue";

// Expose images from socket state
const images = computed(() => socketState.images);

function pauseVideo() {
  socket.emit("pause");
}

function resumeVideo() {
  socket.emit("resume");
}

function resetVideo() {
  socket.emit("reset");
}

function stopVideo() {
  socket.emit("stop");
}
</script>

<template>
  <main>
    <img src="../assets/protostar-logo.png" alt="protostar-logo" />

    <!-- Button row for user controls -->
    <div class="button-row">
      <button @click="pauseVideo">Pause</button>
      <button @click="resumeVideo">Resume</button>
      <button @click="resetVideo">Reset</button>
      <button @click="stopVideo">Stop</button>
    </div>

    <div class="container">
      <div class="camera-container">
        <span class="text-bold">Camera View</span>

        <div
          v-if="images && Object.keys(images).length > 0"
          class="images-container"
        >
          <div v-for="(value, key) in images" :key="key">
            <h3>{{ key }}</h3>
            <div class="image-container">
              <img
                :src="`data:image/jpg;base64,${value}`"
                alt="input"
                class="input-image"
              />
            </div>
          </div>
        </div>

        <div v-else class="image-container">
          <img src="../assets/no-image.png" alt="input is missing" />
          <span>No image available</span>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.images-container {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
}
.container {
  display: flex;
  gap: 13rem;
  margin-top: 85px;
  color: white;
  justify-content: center;
}

.camera-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  justify-content: center;
}

.text-bold {
  font-size: 26px;
  font-weight: 700;
}

.image-container {
  border: 2px solid #44a9b2;
  border-radius: 8px;
  width: 640px;
  aspect-ratio: 16 / 9;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 2rem;
}

.input-image {
  width: 100%;
  height: 100%;
  border-radius: 5px;
}

/* New row for our buttons */
.button-row {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 2rem;
}

/* Responsive styling */
@media (max-width: 1150px) {
  .container {
    gap: 5rem;
  }
}

@media (max-width: 965px) {
  .container {
    align-items: center;
    flex-direction: column;
    gap: 3rem;
  }

  .status-container {
    align-items: center;
  }

  .text-bold {
    margin-left: 0;
    text-align: center;
  }
}
</style>
