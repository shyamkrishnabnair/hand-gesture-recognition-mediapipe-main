const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream);

const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

setInterval(() => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL("image/jpeg");

    fetch("/predict", {
        method: "POST",
        body: JSON.stringify({ image }),
        headers: { "Content-Type": "application/json" }
    }).then(res => res.json()).then(data => {
        console.log("Gesture:", data.gesture);
    });
}, 1000);
