const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const socket = io();
const gestureTableBody = document.getElementById('gestureTableBody');

// socket.on('new_gesture', (gesture) => {
//     const row = document.createElement('tr');
//     row.innerHTML = `
//                 <td class="py-2 px-4 border">${gesture.timestamp}</td>
//                 <td class="py-2 px-4 border">${gesture.handedness}</td>
//                 <td class="py-2 px-4 border">${gesture.type}</td>
//                 <td class="py-2 px-4 border">${gesture.label}</td>
//                 <td class="py-2 px-4 border">${gesture.id}</td>
//             `;
//     gestureTableBody.prepend(row);
// });

async function startRecognition() {
    try {
        const response = await fetch('http://127.0.0.1:5000/start', { method: 'POST' });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();  // Parse the JSON response

        // Handle success or failure based on the status in the JSON
        if (data.status === 'success') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            console.log(data.message);  // Show the message from the backend
        } else {
            console.log(data.message);  // Show the error message
        }
    } catch (error) {
        alert('Error starting recognition: ' + error.message);  // Handle any errors
    }
}

async function stopRecognition() {
    try {
        const response = await fetch('http://127.0.0.1:5000/stop', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            console.log(data.status);
        } else {
            console.log(data.status);
        }
    } catch (error) {
        alert('Error stopping recognition: ' + error.message);
    }
}

// Attach events to buttons
startBtn.addEventListener('click', startRecognition);
stopBtn.addEventListener('click', stopRecognition);