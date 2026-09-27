import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Rectangle,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

type Layer = "fusion" | "insat" | "imerg" | "era5" | "nowcast" | "lis";
type Horizon = 30 | 60 | 90 | 120;

type MultiData = {
  dataset: string;
  times: string[];
  latitude: number[];
  longitude: number[];
  insat: {
    tir1: number[][][];
    tir2: number[][][];
    wv: number[][][];
    vis: number[][][];
    availability: number[][][];
  };
  imerg: {
    precipitation: number[][][];
    availability: number[][][];
  };
  era5: {
    u10: number[][][];
    v10: number[][][];
    d2m: number[][][];
    t2m: number[][][];
    msl: number[][][];
    sp: number[][][];
    tcc: number[][][];
    cape: number[][][];
    availability: number[][][];
  };
  lis: {
    lightning_density: number[][][];
    lightning_availability: number[][][];
    observation_seconds: number[][][];
  };
};

const HORIZONS: Horizon[] = [30, 60, 90, 120];

const layerNames: Record<Layer, string> = {
  fusion: "Fusion",
  insat: "INSAT",
  imerg: "IMERG",
  era5: "ERA5",
  nowcast: "AI Nowcast",
  lis: "LIS Validation",
};

const layerLegend: Record<
  Layer,
  { title: string; left: string; middle: string; right: string }
> = {
  fusion: {
    title: "MULTIMODAL THUNDERSTORM POTENTIAL",
    left: "LOW",
    middle: "MODERATE",
    right: "HIGH",
  },
  insat: {
    title: "INSAT CLOUD-TOP TEMPERATURE",
    left: "COLDER / DEEPER",
    middle: "INTERMEDIATE",
    right: "WARMER",
  },
  imerg: {
    title: "IMERG PRECIPITATION",
    left: "NONE",
    middle: "MODERATE",
    right: "HEAVY",
  },
  era5: {
    title: "ERA5 CAPE / INSTABILITY",
    left: "LOW",
    middle: "MODERATE",
    right: "HIGH",
  },
  nowcast: {
    title: "PROTOTYPE NOWCAST RISK",
    left: "LOW",
    middle: "ELEVATED",
    right: "HIGH",
  },
  lis: {
    title: "LIS OBSERVED LIGHTNING",
    left: "NONE",
    middle: "MODERATE",
    right: "HIGH",
  },
};

function clamp(v: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, v));
}

function normalize(v: number, min: number, max: number) {
  if (!Number.isFinite(v)) return 0;
  return clamp((v - min) / (max - min));
}

function safe(v: unknown, fallback = 0) {
  return Number.isFinite(Number(v)) ? Number(v) : fallback;
}

function formatNumber(v: number, digits = 2) {
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function formatTime(value: string) {
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toISOString().slice(11, 16) + " UTC";
  } catch {
    return value;
  }
}

function getGridValue(
  arr: number[][][] | undefined,
  timeIndex: number,
  y: number,
  x: number
) {
  if (!arr?.[timeIndex]?.[y]) return 0;
  return safe(arr[timeIndex][y][x]);
}

function availability(
  arr: number[][][] | undefined,
  timeIndex: number,
  y: number,
  x: number
) {
  return getGridValue(arr, timeIndex, y, x) > 0 ? 1 : 0;
}

function windSpeed(u: number, v: number) {
  return Math.sqrt(u * u + v * v);
}

function cloudSignal(tir1: number, tir2: number) {
  /*
   * Prototype heuristic only.
   * Lower brightness temperature = colder cloud tops.
   */
  const cold1 = normalize(285 - tir1, 0, 55);
  const cold2 = normalize(285 - tir2, 0, 55);
  return clamp(cold1 * 0.6 + cold2 * 0.4);
}

function precipitationSignal(p: number) {
  return clamp(normalize(p, 0, 10));
}

function capeSignal(cape: number) {
  return clamp(normalize(cape, 0, 2500));
}

function buildFusion(
  data: MultiData,
  timeIndex: number,
  y: number,
  x: number
) {
  const tir1 = getGridValue(data.insat?.tir1, timeIndex, y, x);
  const tir2 = getGridValue(data.insat?.tir2, timeIndex, y, x);
  const wv = getGridValue(data.insat?.wv, timeIndex, y, x);
  const precip = getGridValue(data.imerg?.precipitation, timeIndex, y, x);
  const cape = getGridValue(data.era5?.cape, timeIndex, y, x);
  const tcc = getGridValue(data.era5?.tcc, timeIndex, y, x);
  const u = getGridValue(data.era5?.u10, timeIndex, y, x);
  const v = getGridValue(data.era5?.v10, timeIndex, y, x);

  const sat = cloudSignal(tir1, tir2);
  const rain = precipitationSignal(precip);
  const instability = capeSignal(cape);
  const wind = clamp(normalize(windSpeed(u, v), 0, 20));
  const cloud = clamp(normalize(tcc, 0, 1));

  const satellite = clamp(
    sat * 0.55 +
      normalize(Math.max(0, 285 - wv), 0, 55) * 0.15 +
      cloud * 0.1
  );

  const environment = clamp(
    instability * 0.7 + wind * 0.15 + cloud * 0.15
  );

  const precipGrowth = rain;

  const available =
    availability(data.insat?.availability, timeIndex, y, x) +
    availability(data.imerg?.availability, timeIndex, y, x) +
    availability(data.era5?.availability, timeIndex, y, x);

  const reliability = available / 3;

  const risk = clamp(
    (satellite * 0.35 +
      precipGrowth * 0.25 +
      environment * 0.4) *
      reliability
  );

  return {
    risk,
    reliability,
    satellite,
    environment,
    rain,
    instability,
  };
}

function riskLabel(score: number) {
  if (score >= 0.72) return "High potential";
  if (score >= 0.48) return "Elevated potential";
  if (score >= 0.25) return "Moderate potential";
  return "Low potential";
}

function riskDescription(score: number) {
  if (score >= 0.72)
    return "Several atmospheric signals are strongly aligned with convective development.";
  if (score >= 0.48)
    return "Multiple atmospheric signals support developing convection in this cell.";
  if (score >= 0.25)
    return "Some convective signals are present, but the evidence is mixed.";
  return "Current observations show limited evidence of significant convective development.";
}

function signalState(score: number) {
  if (score >= 0.7) return "HIGH";
  if (score >= 0.45) return "ELEVATED";
  if (score >= 0.2) return "MODERATE";
  return "LOW";
}

function Signal({
  name,
  state,
  children,
}: {
  name: string;
  state: string;
  children: React.ReactNode;
}) {
  return (
    <div className="signal">
      <div className="signalHead">
        <span className="signalName">{name}</span>
        <span className="signalState">{state}</span>
      </div>
      <p>{children}</p>
    </div>
  );
}

function Legend({
  layer,
}: {
  layer: Layer;
}) {
  const l = layerLegend[layer];

  return (
    <div className="mapLegend">
      <div className="legendTitle">{l.title}</div>
      <div className={`legendGradient legend-${layer}`} />
      <div className="legendLabels">
        <span>{l.left}</span>
        <span>{l.middle}</span>
        <span>{l.right}</span>
      </div>
    </div>
  );
}

function MapResize() {
  const map = useMap();

  useEffect(() => {
    const timer = window.setTimeout(() => {
      map.invalidateSize();
    }, 100);

    return () => window.clearTimeout(timer);
  }, [map]);

  return null;
}

export default function App() {
  const [data, setData] = useState<MultiData | null>(null);
  const [error, setError] = useState("");
  const [layer, setLayer] = useState<Layer>("fusion");
  const [timeIndex, setTimeIndex] = useState(1);
  const [horizon, setHorizon] = useState<Horizon>(30);
  const [selected, setSelected] = useState<{
    y: number;
    x: number;
  } | null>(null);

  useEffect(() => {
    let alive = true;

    fetch("/api/multimodal/latest")
      .then((r) => {
        if (!r.ok) throw new Error(`API returned ${r.status}`);
        return r.json();
      })
      .then((json) => {
        if (alive) setData(json);
      })
      .catch((e) => {
        if (alive) setError(String(e));
      });

    return () => {
      alive = false;
    };
  }, []);

  const grid = useMemo(() => {
    if (!data) return [];

    const cells: {
      y: number;
      x: number;
      bounds: [[number, number], [number, number]];
    }[] = [];

    const lat = data.latitude;
    const lon = data.longitude;

    if (lat.length < 2 || lon.length < 2) return [];

    const latStep =
      Math.abs(lat[1] - lat[0]) || 0.03846;

    const lonStep =
      Math.abs(lon[1] - lon[0]) || 0.07692;

    for (let y = 0; y < lat.length; y++) {
      for (let x = 0; x < lon.length; x++) {
        const la = lat[y];
        const lo = lon[x];

        cells.push({
          y,
          x,
          bounds: [
            [la - latStep / 2, lo - lonStep / 2],
            [la + latStep / 2, lo + lonStep / 2],
          ],
        });
      }
    }

    return cells;
  }, [data]);

  const selectedInfo = useMemo(() => {
    if (!data || !selected) return null;

    const { y, x } = selected;

    const tir1 = getGridValue(data.insat?.tir1, timeIndex, y, x);
    const tir2 = getGridValue(data.insat?.tir2, timeIndex, y, x);
    const wv = getGridValue(data.insat?.wv, timeIndex, y, x);
    const vis = getGridValue(data.insat?.vis, timeIndex, y, x);

    const precipitation = getGridValue(
      data.imerg?.precipitation,
      timeIndex,
      y,
      x
    );

    const cape = getGridValue(data.era5?.cape, timeIndex, y, x);
    const t2m = getGridValue(data.era5?.t2m, timeIndex, y, x);
    const d2m = getGridValue(data.era5?.d2m, timeIndex, y, x);
    const u10 = getGridValue(data.era5?.u10, timeIndex, y, x);
    const v10 = getGridValue(data.era5?.v10, timeIndex, y, x);

    const lightning = getGridValue(
      data.lis?.lightning_density,
      timeIndex,
      y,
      x
    );

    const observationSeconds = getGridValue(
      data.lis?.observation_seconds,
      timeIndex,
      y,
      x
    );

    const fusion = buildFusion(data, timeIndex, y, x);

    const decay: Record<Horizon, number> = {
      30: 1,
      60: 0.9,
      90: 0.78,
      120: 0.65,
    };

    const nowcast = clamp(fusion.risk * decay[horizon]);

    const predictiveAvailable =
      availability(data.insat?.availability, timeIndex, y, x) +
      availability(data.imerg?.availability, timeIndex, y, x) +
      availability(data.era5?.availability, timeIndex, y, x);

    return {
      tir1,
      tir2,
      wv,
      vis,
      precipitation,
      cape,
      t2m,
      d2m,
      u10,
      v10,
      wind: windSpeed(u10, v10),
      lightning,
      observationSeconds,
      fusion,
      nowcast,
      predictiveAvailable,
      lat: data.latitude[y],
      lon: data.longitude[x],
      cloud: cloudSignal(tir1, tir2),
      rain: precipitationSignal(precipitation),
      instability: capeSignal(cape),
    };
  }, [data, selected, horizon, timeIndex]);

  const center = useMemo<[number, number]>(() => {
    if (!data || data.latitude.length === 0)
      return [20.5, 87];

    const lat =
      data.latitude.reduce((a, b) => a + b, 0) /
      data.latitude.length;

    const lon =
      data.longitude.reduce((a, b) => a + b, 0) /
      data.longitude.length;

    return [lat, lon];
  }, [data]);

  function cellValue(y: number, x: number) {
    if (!data) return 0;

    const fusion = buildFusion(data, timeIndex, y, x);

    switch (layer) {
      case "fusion":
        return fusion.risk;

      case "insat":
        return cloudSignal(
          getGridValue(data.insat.tir1, timeIndex, y, x),
          getGridValue(data.insat.tir2, timeIndex, y, x)
        );

      case "imerg":
        return precipitationSignal(
          getGridValue(data.imerg.precipitation, timeIndex, y, x)
        );

      case "era5":
        return capeSignal(
          getGridValue(data.era5.cape, timeIndex, y, x)
        );

      case "nowcast":
        return fusion.risk *
          ({
            30: 1,
            60: 0.9,
            90: 0.78,
            120: 0.65,
          } as Record<Horizon, number>)[horizon];

      case "lis":
        return normalize(
          getGridValue(
            data.lis.lightning_density,
            timeIndex,
            y,
            x
          ),
          0,
          1
        );

      default:
        return 0;
    }
  }

  if (error) {
    return (
      <div className="appError">
        <strong>Multimodal API connection failed</strong>
        <p>{error}</p>
        <p>
          Make sure the FastAPI backend is running on port 8010.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="loadingScreen">
        <div className="loadingDot" />
        Loading multimodal observations…
      </div>
    );
  }

  return (
    <div className="app">
      <style>{`
        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          background: #091013;
          color: #eef3f4;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .app {
          height: 100vh;
          min-height: 620px;
          display: grid;
          grid-template-rows: 58px minmax(0, 1fr) 74px;
          background: #091013;
          overflow: hidden;
        }

        .topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 20px;
          border-bottom: 1px solid rgba(255,255,255,.08);
          background: rgba(9,16,19,.96);
          z-index: 1000;
        }

        .brand {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .brandMark {
          width: 28px;
          height: 28px;
          border: 1px solid rgba(255,255,255,.25);
          border-radius: 7px;
          display: grid;
          place-items: center;
          font-size: 12px;
          font-weight: 800;
        }

        .brandTitle {
          font-size: 13px;
          font-weight: 700;
          letter-spacing: .08em;
        }

        .brandSub {
          font-size: 9px;
          color: #839096;
          letter-spacing: .1em;
          margin-top: 2px;
        }

        .eventMeta {
          text-align: right;
          font-size: 10px;
          color: #839096;
          line-height: 1.5;
        }

        .eventMeta strong {
          color: #e8eeee;
          font-weight: 600;
        }

        .main {
          min-height: 0;
          display: grid;
          grid-template-columns: 1fr 355px;
        }

        .mapArea {
          position: relative;
          min-width: 0;
          min-height: 0;
          background: #11191d;
        }

        .map {
          width: 100%;
          height: 100%;
        }

        .mapHeader {
          position: absolute;
          left: 18px;
          top: 16px;
          z-index: 800;
          pointer-events: none;
        }

        .mapEyebrow {
          font-size: 9px;
          letter-spacing: .12em;
          color: #8e9ba0;
        }

        .mapTitle {
          font-size: 17px;
          font-weight: 650;
          margin-top: 4px;
          text-shadow: 0 1px 8px rgba(0,0,0,.8);
        }

        .layerRail {
          position: absolute;
          left: 18px;
          top: 82px;
          z-index: 800;
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .layerButton {
          appearance: none;
          border: 1px solid rgba(255,255,255,.12);
          background: rgba(8,14,17,.88);
          color: #879398;
          padding: 8px 11px;
          min-width: 116px;
          border-radius: 6px;
          text-align: left;
          font-size: 9px;
          letter-spacing: .09em;
          cursor: pointer;
          backdrop-filter: blur(8px);
        }

        .layerButton:hover {
          color: #fff;
          border-color: rgba(255,255,255,.25);
        }

        .layerButton.active {
          color: #fff;
          border-color: rgba(255,214,70,.65);
          background: rgba(75,65,16,.72);
        }

        .mapLegend {
          position: absolute;
          right: 18px;
          bottom: 16px;
          z-index: 800;
          width: 215px;
          padding: 11px 12px;
          border: 1px solid rgba(255,255,255,.12);
          background: rgba(8,14,17,.9);
          border-radius: 7px;
          backdrop-filter: blur(8px);
        }

        .legendTitle {
          color: #b7c0c3;
          font-size: 8px;
          letter-spacing: .1em;
          margin-bottom: 8px;
        }

        .legendGradient {
          height: 8px;
          border-radius: 5px;
          background: linear-gradient(90deg,#123c4a,#20a37a,#c9dc43,#ffb52b,#e43d3d);
        }

        .legend-insat {
          background: linear-gradient(90deg,#e43d3d,#ff9c32,#d8d443,#4ba890,#183e4b);
        }

        .legend-imerg {
          background: linear-gradient(90deg,#182d3a,#287f76,#9ed044,#ffb52b,#e43d3d);
        }

        .legend-era5 {
          background: linear-gradient(90deg,#173744,#277f70,#c9dc43,#ff9f2f,#e43d3d);
        }

        .legend-nowcast {
          background: linear-gradient(90deg,#173744,#287f76,#c9dc43,#ff9f2f,#e43d3d);
        }

        .legend-lis {
          background: linear-gradient(90deg,#16252d,#286e83,#45a9c5,#d9dc50,#ff773d);
        }

        .legendLabels {
          display: flex;
          justify-content: space-between;
          color: #78858a;
          font-size: 7px;
          margin-top: 5px;
          gap: 6px;
        }

        .inspector {
          min-width: 0;
          overflow-y: auto;
          background: #0d1518;
          border-left: 1px solid rgba(255,255,255,.08);
          padding: 18px;
        }

        .inspector::-webkit-scrollbar {
          width: 5px;
        }

        .inspector::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,.12);
          border-radius: 5px;
        }

        .cellEyebrow {
          color: #77858a;
          font-size: 8px;
          letter-spacing: .12em;
        }

        .cellTitle {
          font-size: 18px;
          font-weight: 650;
          margin-top: 5px;
        }

        .availability {
          display: flex;
          align-items: center;
          gap: 7px;
          color: #89969a;
          font-size: 9px;
          margin-top: 7px;
        }

        .availabilityDot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #7bd65c;
        }

        .assessment {
          margin-top: 17px;
          padding: 15px;
          border-radius: 9px;
          border: 1px solid rgba(255,165,40,.3);
          background: rgba(255,130,20,.065);
        }

        .assessmentLabel {
          color: #88959a;
          font-size: 8px;
          letter-spacing: .12em;
        }

        .assessmentValue {
          font-size: 20px;
          font-weight: 650;
          margin-top: 5px;
        }

        .assessmentText {
          color: #9aa6aa;
          font-size: 10px;
          line-height: 1.5;
          margin: 7px 0 0;
        }

        .forecastRow {
          margin-top: 13px;
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 4px;
        }

        .forecastButton {
          appearance: none;
          border: 1px solid rgba(255,255,255,.1);
          background: #121c20;
          color: #77858a;
          padding: 7px 3px;
          border-radius: 5px;
          cursor: pointer;
          font-size: 8px;
        }

        .forecastButton.active {
          color: #fff;
          border-color: rgba(255,214,70,.55);
          background: rgba(100,86,18,.32);
        }

        .forecastButton strong {
          display: block;
          font-size: 10px;
          margin-bottom: 2px;
        }

        .signalSection {
          margin-top: 15px;
        }

        .sectionTitle {
          color: #77858a;
          font-size: 8px;
          letter-spacing: .12em;
          margin-bottom: 2px;
        }

        .signal {
          padding: 11px 0;
          border-bottom: 1px solid rgba(255,255,255,.07);
        }

        .signalHead {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 8px;
        }

        .signalName {
          font-size: 11px;
          font-weight: 600;
        }

        .signalState {
          color: #b8dc62;
          background: rgba(119,190,70,.1);
          border: 1px solid rgba(119,190,70,.16);
          border-radius: 4px;
          padding: 3px 5px;
          font-size: 7px;
          letter-spacing: .08em;
        }

        .signal p {
          color: #879398;
          font-size: 9px;
          line-height: 1.45;
          margin: 5px 0 0;
        }

        .evidence {
          margin-top: 15px;
          padding-top: 13px;
          border-top: 1px solid rgba(255,255,255,.08);
        }

        .evidenceGrid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 5px;
          margin-top: 8px;
        }

        .evidenceItem {
          background: #111b1f;
          border: 1px solid rgba(255,255,255,.06);
          border-radius: 6px;
          padding: 8px;
        }

        .evidenceItem span {
          display: block;
          color: #68767b;
          font-size: 7px;
          letter-spacing: .06em;
        }

        .evidenceItem strong {
          display: block;
          margin-top: 3px;
          font-size: 10px;
          font-weight: 600;
        }

        .validation {
          margin-top: 12px;
          padding: 10px;
          border-radius: 7px;
          background: rgba(60,120,150,.07);
          border: 1px solid rgba(80,150,180,.12);
        }

        .validationTitle {
          font-size: 8px;
          letter-spacing: .1em;
          color: #829197;
        }

        .validationText {
          color: #8c999d;
          font-size: 9px;
          line-height: 1.45;
          margin-top: 5px;
        }

        .rawNote {
          color: #606d71;
          font-size: 8px;
          line-height: 1.4;
          margin-top: 8px;
        }

        .timeline {
          border-top: 1px solid rgba(255,255,255,.08);
          background: #0a1114;
          padding: 11px 20px 10px;
          z-index: 1000;
        }

        .timelineTop {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .timelineLabel {
          color: #77858a;
          font-size: 8px;
          letter-spacing: .1em;
        }

        .timelineCurrent {
          color: #dce3e5;
          font-size: 9px;
        }

        .timelineTrack {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 5px;
        }

        .timeButton {
          appearance: none;
          border: 0;
          border-top: 2px solid #253238;
          background: transparent;
          color: #677479;
          padding: 6px 2px;
          cursor: pointer;
          font-size: 8px;
          text-align: left;
        }

        .timeButton.active {
          border-top-color: #d9c454;
          color: #fff;
        }

        .timeButton span {
          display: block;
          margin-top: 4px;
        }

        .loadingScreen,
        .appError {
          min-height: 100vh;
          display: grid;
          place-content: center;
          text-align: center;
          background: #091013;
          color: #dfe7e8;
          padding: 30px;
        }

        .loadingDot {
          width: 10px;
          height: 10px;
          margin: 0 auto 12px;
          border-radius: 50%;
          background: #d8c452;
        }

        .appError p {
          color: #849095;
          font-size: 12px;
        }

        .leaflet-container {
          background: #11191d;
          font-family: inherit;
        }

        @media (max-width: 900px) {
          .main {
            grid-template-columns: 1fr;
            grid-template-rows: minmax(420px, 1fr) auto;
            overflow-y: auto;
          }

          .mapArea {
            min-height: 500px;
          }

          .inspector {
            max-height: 540px;
          }

          .app {
            height: auto;
            min-height: 100vh;
            grid-template-rows: 58px auto 74px;
          }
        }

        @media (max-width: 600px) {
          .eventMeta {
            display: none;
          }

          .layerRail {
            left: 10px;
            top: 75px;
          }

          .mapLegend {
            right: 10px;
            bottom: 10px;
            width: 175px;
          }

          .mapHeader {
            left: 10px;
          }

          .timeline {
            padding-left: 10px;
            padding-right: 10px;
          }
        }
      `}</style>

      <header className="topbar">
        <div className="brand">
          <div className="brandMark">TS</div>
          <div>
            <div className="brandTitle">
              THUNDERSTORM NOWCAST
            </div>
            <div className="brandSub">
              MULTIMODAL ATMOSPHERIC INTELLIGENCE
            </div>
          </div>
        </div>

        <div className="eventMeta">
          <strong>REAL OBSERVATIONS</strong>
          <br />
          OBSERVATION ? {formatTime(data.times[timeIndex] ?? data.times[0])}
          <br />
          {data.dataset}
        </div>
      </header>

      <main className="main">
        <section className="mapArea">
          <div className="map">
            <MapContainer
              center={center}
              zoom={8}
              minZoom={6}
              maxZoom={13}
              zoomControl={true}
              attributionControl={true}
              style={{ width: "100%", height: "100%" }}
            >
              <MapResize />

              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              {grid.map((cell) => {
                const value = cellValue(cell.y, cell.x);

                const opacity =
                  layer === "lis"
                    ? 0.15 + value * 0.7
                    : 0.12 + value * 0.65;

                const hue =
                  value < 0.25
                    ? 165
                    : value < 0.5
                    ? 75
                    : value < 0.72
                    ? 35
                    : 5;

                const fillColor =
                  `hsl(${hue}, ${layer === "insat" ? 65 : 75}%, ${layer === "insat" ? 46 : 48}%)`;

                const selectedCell =
                  selected?.y === cell.y &&
                  selected?.x === cell.x;

                return (
                  <Rectangle
                    key={`${cell.y}-${cell.x}`}
                    bounds={cell.bounds}
                    pathOptions={{
                      color: selectedCell
                        ? "#ffffff"
                        : "rgba(255,255,255,.16)",
                      weight: selectedCell ? 2 : 0.5,
                      fillColor,
                      fillOpacity: opacity,
                    }}
                    eventHandlers={{
                      click: () =>
                        setSelected({
                          y: cell.y,
                          x: cell.x,
                        }),
                    }}
                  />
                );
              })}
            </MapContainer>
          </div>

          <div className="mapHeader">
            <div className="mapEyebrow">
              REAL MULTIMODAL EVENT · 01 MAY 2020
            </div>
            <div className="mapTitle">
              Odisha · Common 27 × 27 Grid
            </div>
          </div>

          <div className="layerRail">
            {(Object.keys(layerNames) as Layer[]).map((item) => (
              <button
                key={item}
                type="button"
                className={`layerButton ${
                  layer === item ? "active" : ""
                }`}
                onClick={() => setLayer(item)}
              >
                {layerNames[item].toUpperCase()}
              </button>
            ))}
          </div>

          <Legend layer={layer} />
        </section>

        <aside className="inspector">
          {selectedInfo ? (
            <>
              <div className="cellEyebrow">
                SELECTED CELL · {selectedInfo.lat.toFixed(2)}°N ·{" "}
                {selectedInfo.lon.toFixed(2)}°E
              </div>

              <div className="cellTitle">
                Thunderstorm Intelligence
              </div>

              <div className="availability">
                <span className="availabilityDot" />
                {selectedInfo.predictiveAvailable === 3
                  ? "All predictive sources available"
                  : `${selectedInfo.predictiveAvailable}/3 predictive sources available`}
              </div>

              <div className="assessment">
                <div className="assessmentLabel">
                  CURRENT MULTIMODAL ASSESSMENT
                </div>

                <div className="assessmentValue">
                  {riskLabel(selectedInfo.fusion.risk)}
                </div>

                <p className="assessmentText">
                  {riskDescription(selectedInfo.fusion.risk)}
                </p>

                <div className="forecastRow">
                  {HORIZONS.map((h) => (
                    <button
                      key={h}
                      type="button"
                      className={`forecastButton ${
                        horizon === h ? "active" : ""
                      }`}
                      onClick={() => setHorizon(h)}
                    >
                      <strong>+{h}</strong>
                      min
                    </button>
                  ))}
                </div>
              </div>

              <div className="signalSection">
                <div className="sectionTitle">
                  WHY THE SYSTEM SAYS THIS
                </div>

                <Signal
                  name="Cloud development"
                  state={signalState(selectedInfo.cloud)}
                >
                  {selectedInfo.tir1 < 260
                    ? "INSAT shows cold cloud-top temperatures, a signal consistent with deep or developing convection."
                    : selectedInfo.tir1 < 275
                    ? "INSAT shows moderately cold cloud tops. Some convective development is indicated."
                    : "INSAT cloud tops are relatively warm, giving limited evidence of deep convection."}
                </Signal>

                <Signal
                  name="Atmospheric instability"
                  state={signalState(selectedInfo.instability)}
                >
                  {selectedInfo.cape >= 1500
                    ? `ERA5 CAPE is ${formatNumber(
                        selectedInfo.cape,
                        0
                      )} J/kg, indicating a strongly unstable convective environment.`
                    : selectedInfo.cape >= 700
                    ? `ERA5 CAPE is ${formatNumber(
                        selectedInfo.cape,
                        0
                      )} J/kg, indicating moderate instability.`
                    : `ERA5 CAPE is ${formatNumber(
                        selectedInfo.cape,
                        0
                      )} J/kg, indicating limited atmospheric instability.`}
                </Signal>

                <Signal
                  name="Precipitation activity"
                  state={
                    selectedInfo.precipitation > 5
                      ? "HIGH"
                      : selectedInfo.precipitation > 0.5
                      ? "PRESENT"
                      : "LOW"
                  }
                >
                  {selectedInfo.precipitation > 5
                    ? `IMERG indicates substantial precipitation at approximately ${formatNumber(
                        selectedInfo.precipitation,
                        2
                      )}.`
                    : selectedInfo.precipitation > 0.5
                    ? `IMERG indicates precipitation is present at approximately ${formatNumber(
                        selectedInfo.precipitation,
                        2
                      )}.`
                    : "IMERG shows little or no precipitation in this cell."}
                </Signal>

                <Signal
                  name="Low-level wind environment"
                  state={
                    selectedInfo.wind >= 12
                      ? "STRONG"
                      : selectedInfo.wind >= 6
                      ? "MODERATE"
                      : "LIGHT"
                  }
                >
                  {`ERA5 10-m winds are approximately ${formatNumber(
                    selectedInfo.wind,
                    1
                  )} m/s. This describes the local low-level wind environment used by the prototype.`
                  }
                </Signal>

                <Signal
                  name="Observation reliability"
                  state={
                    selectedInfo.fusion.reliability >= 0.99
                      ? "HIGH"
                      : selectedInfo.fusion.reliability >= 0.66
                      ? "PARTIAL"
                      : "LOW"
                  }
                >
                  {selectedInfo.predictiveAvailable === 3
                    ? "INSAT, IMERG and ERA5 observations are available for this cell."
                    : "One or more predictive sources are unavailable, so the fusion score is reduced for reliability."}
                </Signal>
              </div>

              <div className="evidence">
                <div className="sectionTitle">
                  SUPPORTING OBSERVATIONS
                </div>

                <div className="evidenceGrid">
                  <div className="evidenceItem">
                    <span>INSAT TIR1</span>
                    <strong>
                      {formatNumber(selectedInfo.tir1, 1)} K
                    </strong>
                  </div>

                  <div className="evidenceItem">
                    <span>INSAT TIR2</span>
                    <strong>
                      {formatNumber(selectedInfo.tir2, 1)} K
                    </strong>
                  </div>

                  <div className="evidenceItem">
                    <span>IMERG</span>
                    <strong>
                      {formatNumber(
                        selectedInfo.precipitation,
                        2
                      )}
                    </strong>
                  </div>

                  <div className="evidenceItem">
                    <span>ERA5 CAPE</span>
                    <strong>
                      {formatNumber(selectedInfo.cape, 0)} J/kg
                    </strong>
                  </div>

                  <div className="evidenceItem">
                    <span>T2M</span>
                    <strong>
                      {formatNumber(selectedInfo.t2m, 1)} K
                    </strong>
                  </div>

                  <div className="evidenceItem">
                    <span>10-M WIND</span>
                    <strong>
                      {formatNumber(selectedInfo.wind, 1)} m/s
                    </strong>
                  </div>
                </div>

                <div className="validation">
                  <div className="validationTitle">
                    LIS VALIDATION
                  </div>

                  <div className="validationText">
                    {selectedInfo.lightning > 0
                      ? `Observed lightning is present in this cell. LIS recorded approximately ${formatNumber(
                          selectedInfo.lightning,
                          3
                        )} lightning density units during an available observation window of ${formatNumber(
                          selectedInfo.observationSeconds,
                          0
                        )} seconds.`
                      : selectedInfo.observationSeconds > 0
                      ? "LIS observation was available for this cell, but no lightning was recorded."
                      : "No LIS observation is available for this cell at this time."}
                  </div>
                </div>

                <div className="rawNote">
                  Scientific values are retained as evidence. The
                  interpretation layer above converts those measurements
                  into human-readable atmospheric signals. Current fusion
                  and nowcast values are prototype scores, not calibrated
                  probabilities.
                </div>
              </div>
            </>
          ) : (
            <div style={{ paddingTop: 80, textAlign: "center" }}>
              <div
                style={{
                  color: "#dfe6e8",
                  fontSize: 15,
                  fontWeight: 600,
                }}
              >
                Select a grid cell
              </div>
              <div
                style={{
                  color: "#758287",
                  fontSize: 10,
                  lineHeight: 1.5,
                  marginTop: 8,
                }}
              >
                Click any cell on the multimodal map to inspect its
                atmospheric signals and interpretation.
              </div>
            </div>
          )}
        </aside>
      </main>

      <footer className="timeline">
        <div className="timelineTop">
          <div className="timelineLabel">
            OBSERVATION TIMELINE · MULTIMODAL DATA
          </div>

          <div className="timelineCurrent">
            {formatTime(data.times[timeIndex] ?? data.times[0])}
          </div>
        </div>

        <div className="timelineTrack">
          {data.times.map((time, index) => (
            <button
              type="button"
              key={`${time}-${index}`}
              className={`timeButton ${
                timeIndex === index ? "active" : ""
              }`}
              onClick={() => setTimeIndex(index)}
            >
              OBSERVATION
              <span>{formatTime(time)}</span>
            </button>
          ))}
        </div>
      </footer>
    </div>
  );
}

