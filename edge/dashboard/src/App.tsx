import { useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { LogsDrawer } from "./components/LogsDrawer";
import { useTelemetryStream } from "./hooks/useTelemetryStream";
import { BubblePage } from "./pages/BubblePage";
import { SettingsPage } from "./pages/SettingsPage";
import { TrialsPage } from "./pages/TrialsPage";

export default function App() {
  const { link, telemetry, trialsSource, lastError } = useTelemetryStream();
  const [logsOpen, setLogsOpen] = useState(false);

  return (
    <BrowserRouter>
      <Routes>
        <Route
          element={
            <AppShell
              link={link}
              telemetry={telemetry}
              trialsSource={trialsSource}
              lastError={lastError}
              onOpenLogs={() => setLogsOpen(true)}
            />
          }
        >
          <Route index element={<Navigate to="/bubble" replace />} />
          <Route path="bubble" element={<BubblePage telemetry={telemetry} />} />
          <Route
            path="trials"
            element={<TrialsPage telemetry={telemetry} trialsSource={trialsSource} />}
          />
          <Route path="settings" element={<SettingsPage telemetry={telemetry} />} />
          <Route path="*" element={<Navigate to="/bubble" replace />} />
        </Route>
      </Routes>
      <LogsDrawer open={logsOpen} onClose={() => setLogsOpen(false)} />
    </BrowserRouter>
  );
}
