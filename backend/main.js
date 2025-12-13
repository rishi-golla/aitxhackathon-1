const { app, BrowserWindow } = require("electron")

let win

app.whenReady().then(() => {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  win.loadURL("http://localhost:3000/login")
  
  // Open DevTools in development
  if (process.env.NODE_ENV !== "production") {
    win.webContents.openDevTools()
  }
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit()
  }
})

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    win = new BrowserWindow({
      width: 1400,
      height: 900,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
      },
    })
    win.loadURL("http://localhost:3000/login")
  }
})
