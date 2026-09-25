
import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Rectangle,
  Popup,
  useMap,
} from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

type InsatResponse = {
  filename: string;
  timestamp: string;
  channels: string[];
  shape: number[];
  latitude: number[][];
  longitude: number[][];
  data: number[][][];
};

type GridCell = {
  id: string;
  row: number;
  col: number;
  bounds: LatLngBoundsExpression;
  tir1: number;
  tir2: number;
  mir: number;
  wv: number;
};

type LayerType = "thunderstorm" | "insat";

function FitMap({ bounds }: { bounds: LatLngBoundsExpression }) {
  const map = useMap();

  useEffect(() => {
    map.fitBounds(bounds, {
      padding: [30, 30],
    });
  }, [map, bounds]);

  return null;
}

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function getTemperatureColor(value: number) {
  if (value < 230) return "#ff1744";
  if (value < 250) return "#ff5252";
  if (value < 270) return "#ff9800";
  if (value < 280) return "#ffd54f";
  if (value < 290) return "#66bb6a";
  if (value < 300) return "#29b6f6";
  return "#42a5f5";
}

function getStormColor(value: number) {
  if (value >= 0.8) return "#ff1744";
  if (value >= 0.6) return "#ff6d00";
  if (value >= 0.4) return "#ffd600";
  if (value >= 0.2) return "#76ff03";
  return "#00e676";
}

export default function App() {
  const [insat, setInsat] = useState<InsatResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [layer, setLayer] = useState<LayerType>("insat");
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [forecast, setForecast] = useState(30);

  useEffect(() => {
   fetch(`${import.meta.env.VITE_API_URL}/api/insat/latest`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend request failed");
        }

        return response.json();
      })
      .then((result: InsatResponse) => {
        setInsat(result);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const gridCells = useMemo(() => {
    if (!insat) {
      return [];
    }

    const rows = insat.latitude.length;
    const cols = insat.latitude[0].length;

    const cells: GridCell[] = [];

    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const lat1 = insat.latitude[row][col];
        const lon1 = insat.longitude[row][col];

        const lat2 = insat.latitude[row + 1][col + 1];
        const lon2 = insat.longitude[row + 1][col + 1];

        const tir1 = insat.data[row][col][1];
        const tir2 = insat.data[row][col][2];
        const mir = insat.data[row][col][0];
        const wv = insat.data[row][col][3];

        const bounds: LatLngBoundsExpression = [
          [Math.min(lat1, lat2), Math.min(lon1, lon2)],
          [Math.max(lat1, lat2), Math.max(lon1, lon2)],
        ];

        cells.push({
          id: String(row) + "-" + String(col),
          row,
          col,
          bounds,
          tir1,
          tir2,
          mir,
          wv,
        });
      }
    }

    return cells;
  }, [insat]);

  const mapBounds = useMemo(() => {
    if (!insat) {
      return null;
    }

    const lats = insat.latitude.flat();
    const lons = insat.longitude.flat();

    return [
      [Math.min(...lats), Math.min(...lons)],
      [Math.max(...lats), Math.max(...lons)],
    ] as LatLngBoundsExpression;
  }, [insat]);

  if (loading) {
    return (
      <div className="app-loading">
        <div className="loading-title">THUNDERSTORM NOWCASTING</div>
        <div className="loading-subtitle">
          Loading INSAT observations...
        </div>
      </div>
    );
  }

  if (error || !insat || !mapBounds) {
    return (
      <div className="app-loading">
        <div className="loading-title">DATA CONNECTION ERROR</div>
        <div className="loading-subtitle">
          {error || "No INSAT data available"}
        </div>
        <div className="loading-help">
          Make sure the FastAPI backend is running on port 8000.
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="brand">THUNDERSTORM</div>
          <div className="brand-subtitle">NOWCASTING SYSTEM</div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          SYSTEM ONLINE
        </div>

        <div className="header-time">
          OBSERVATION
          <strong>{formatTime(insat.timestamp)}</strong>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="panel-title">DATA LAYERS</div>

          <button
            className={
              layer === "thunderstorm"
                ? "layer-button active"
                : "layer-button"
            }
            onClick={() => setLayer("thunderstorm")}
          >
            <span className="layer-icon storm-icon">⚡</span>
            <span>
              <strong>Storm Indicator</strong>
              <small>TIR1-derived indicator</small>
            </span>
          </button>

          <button
            className={
              layer === "insat" ? "layer-button active" : "layer-button"
            }
            onClick={() => setLayer("insat")}
          >
            <span className="layer-icon satellite-icon">◉</span>
            <span>
              <strong>INSAT Satellite</strong>
              <small>Real observation</small>
            </span>
          </button>

          <div className="panel-divider" />

          <div className="panel-title">FORECAST</div>

          {[30, 60, 90, 120].map((minutes) => (
            <button
              key={minutes}
              className={
                forecast === minutes
                  ? "forecast-button selected"
                  : "forecast-button"
              }
              onClick={() => setForecast(minutes)}
            >
              +{minutes} MIN
            </button>
          ))}

          <div className="panel-divider" />

          <div className="panel-title">DATA STATUS</div>

          <div className="data-status">
            <div>
              <span className="status-dot green" />
              INSAT
              <strong>AVAILABLE</strong>
            </div>

            <div>
              <span className="status-dot gray" />
              RADAR
              <strong>OFFLINE</strong>
            </div>

            <div>
              <span className="status-dot gray" />
              LIGHTNING
              <strong>OFFLINE</strong>
            </div>

            <div>
              <span className="status-dot gray" />
              AWS
              <strong>OFFLINE</strong>
            </div>
          </div>
        </aside>

        <main className="map-area">
          <div className="map-overlay-title">
            <div className="location-name">MUMBAI REGION</div>
            <div className="location-subtitle">
              INSAT 3SIMG • 27 × 27 GRID • REAL OBSERVATION
            </div>
          </div>

          <MapContainer
            className="weather-map"
            center={[19.076, 72.8777]}
            zoom={8}
            scrollWheelZoom={true}
          >
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <FitMap bounds={mapBounds} />

            {gridCells.map((cell) => {
              let fillColor = "#263238";
              let fillOpacity = 0.25;

              if (layer === "insat") {
                fillColor = getTemperatureColor(cell.tir1);
                fillOpacity = 0.55;
              }

              if (layer === "thunderstorm") {
                const risk = Math.max(
                  0,
                  Math.min(1, (285 - cell.tir1) / 35)
                );

                fillColor = getStormColor(risk);
                fillOpacity = 0.55;
              }

              return (
                <Rectangle
                  key={cell.id}
                  bounds={cell.bounds}
                  pathOptions={{
                    color: "rgba(255,255,255,0.18)",
                    weight: 0.5,
                    fillColor,
                    fillOpacity,
                  }}
                  eventHandlers={{
                    click: () => setSelectedCell(cell),
                  }}
                >
                  <Popup>
                    <div className="popup">
                      <strong>
                        GRID {cell.row} / {cell.col}
                      </strong>
                      <br />
                      TIR1: {cell.tir1.toFixed(2)} K
                      <br />
                      TIR2: {cell.tir2.toFixed(2)} K
                      <br />
                      MIR: {cell.mir.toFixed(2)} K
                      <br />
                      WV: {cell.wv.toFixed(2)} K
                    </div>
                  </Popup>
                </Rectangle>
              );
            })}
          </MapContainer>

          <div className="map-legend">
            <div className="legend-title">
              {layer === "insat"
                ? "TIR1 TEMPERATURE"
                : "STORM INDICATOR"}
            </div>

            <div className="legend-gradient" />

            <div className="legend-labels">
              <span>HIGH</span>
              <span>MEDIUM</span>
              <span>LOW</span>
            </div>
          </div>
        </main>

        <aside className="right-panel">
          <div className="panel-title">SELECTED CELL</div>

          {selectedCell ? (
            <>
              <div className="cell-id">
                GRID {selectedCell.row} / {selectedCell.col}
              </div>

              <div className="risk-card">
                <div className="risk-label">FORECAST WINDOW</div>
                <div className="risk-value">+{forecast} MIN</div>
              </div>

              <div className="metric">
                <span>TIR1</span>
                <strong>{selectedCell.tir1.toFixed(2)} K</strong>
              </div>

              <div className="metric">
                <span>TIR2</span>
                <strong>{selectedCell.tir2.toFixed(2)} K</strong>
              </div>

              <div className="metric">
                <span>MIR</span>
                <strong>{selectedCell.mir.toFixed(2)} K</strong>
              </div>

              <div className="metric">
                <span>WV</span>
                <strong>{selectedCell.wv.toFixed(2)} K</strong>
              </div>

              <div className="panel-divider" />

              <div className="panel-title">MODEL STATUS</div>

              <div className="model-status">
                <div>
                  <span className="status-dot yellow" />
                  AI MODEL
                </div>

                <small>
                  Prototype model connected.
                  <br />
                  Radar and lightning inputs are
                  currently unavailable.
                </small>
              </div>
            </>
          ) : (
            <div className="no-selection">
              <div className="crosshair">＋</div>
              <p>Select a grid cell</p>
              <small>
                Click any INSAT observation cell on the map to inspect its
                atmospheric values.
              </small>
            </div>
          )}
        </aside>
      </div>

      <footer className="bottom-bar">
        <div>
          <span className="footer-label">SOURCE</span>
          INSAT-3SIMG / MOSDAC
        </div>

        <div>
          <span className="footer-label">TIMESTAMP</span>
          {insat.timestamp}
        </div>

        <div>
          <span className="footer-label">GRID</span>
          {insat.shape[0]} × {insat.shape[1]}
        </div>

        <div>
          <span className="footer-label">CHANNELS</span>
          {insat.channels.join(" / ")}
        </div>
      </footer>
    </div>
  );
}
