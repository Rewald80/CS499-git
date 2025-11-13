const path = require("path");
const { spawn } = require("child_process");

const projectDir = "C:\\Users\\rewald\\OneDrive\\Desktop\\Semo\\Fall 25\\CS499\\CS499-git\\detector";
function startServer() {
  const server = spawn("python", ["flwr_server.py"], {cwd: projectDir });
  server.stdout.on("data", (data) => console.log(`[SERVER] ${data}`));
  server.stderr.on("data", (data) => console.error(`[SERVER ERROR] ${data}`));
}

function startClient() {
  const client = spawn("python", ["flwr_client.py"], { cwd: projectDir});
  client.stdout.on("data", (data) => console.log(`[CLIENT] ${data}`));
  client.stderr.on("data", (data) => console.error(`[CLIENT ERROR] ${data}`));
}

startServer();
setTimeout(startClient, 2000);