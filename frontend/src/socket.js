import { reactive } from "vue";
import { io } from "socket.io-client";

export const state = reactive({
  connected: false,
  image: null,
});

const URL = "http://localhost:8000";

export const socket = io(URL);

socket.on("connect", () => {
  state.connected = true;
});

socket.on("disconnect", () => {
  state.connected = false;
});

socket.on("image", (data) => {
  state.image = data.image;
});
