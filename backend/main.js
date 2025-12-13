const { app, BrowserWindow } = require("electron")
const path = require("path")

let win

app.whenReady().then(() => {
  win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  })

  win.loadURL("http://localhost:3000")
})
