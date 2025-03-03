<script setup>
import { state as socketState } from "@/socket";
import { computed, ref } from "vue";

const image = computed(() => socketState.image);
const angle = computed(() => {
  return socketState.angle;
});

</script>

<template>
  <main>
    <img src="../assets/protostar-logo.png" alt="protostar-logo" />
    <div class="container">
      <div class="camera-container">
        <span class="text-bold">Camera View</span>

        <div v-if="image === 'unprocessable'" class="image-container">
          <img src="../assets/warning.png" alt="warning icon" />
          <span>Not processable image</span>
        </div>

        <div v-else-if="image" class="image-container">
          <img
            :src="`data:image/jpg;base64,${image}`"
            alt="input"
            class="input-image"
          />
        </div>

        <div v-else class="image-container">
          <img src="../assets/no-image.png" alt="input is missing" />
          <span>No image available</span>
        </div>

        <span v-if="angle" class="angle-text"
          >Angle: <span class="old-font">{{ angle }}</span></span
        >
      </div>
      <div class="status-and-graph-container">
        <div class="status-container">
          <span class="text-bold">Status</span>
          
          <span class="text-bold history-text">History</span>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
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

.status-and-graph-container {
  display: flex;
  flex-direction: column;
  gap: 5rem;
}

.status-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  align-items: flex-start;
}

.image-container {
  border: 2px solid #44a9b2;
  border-radius: 8px;
  width: 378px;
  aspect-ratio: 1 / 1;
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

.history-text {
  margin-top: 2rem;
}

.angle-text {
  font-size: 1.2rem;
  align-self: center;
}

.old-font {
  font-family: sans-serif;
}

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

  .status-and-graph-container {
    align-items: center;
  }
}
</style>
