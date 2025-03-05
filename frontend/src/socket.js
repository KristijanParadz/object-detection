import { reactive } from "vue";
import { io } from "socket.io-client";

export const state = reactive({
  connected: false,
  images: {},
});

const URL = import.meta.env.VITE_API_BASE_URL;

export const socket = io(URL, {
  path: "/socket.io",
});

socket.on("connect", () => {
  socket.emit("start");
  state.connected = true;
});

socket.on("disconnect", () => {
  state.connected = false;
});

socket.on("image", (data) => {
  state.images[data.video_id] = data.image;
});
