async function startTraining() {
    const response = await fetch("http://127.0.0.1:5000/start_flwr", { method: "POST"});
    const data = await response.json();
    console.log("Server response:", data);
}

async function getStatus() {
    const response = await fetch("http://127.0.0.1:5000/status");
    const data = await response.json();
    console.log("Current training status:", data);
}

startTraining();