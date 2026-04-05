/**
 * TRIBEv2 — Cinematic Analysis HUD
 * Fixes:
 *  - Face graph has a labeled legend
 *  - Semantic / audio panels persist last known values (dimmed) during silence/gaps
 *  - No jarring blank states anywhere
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip as ReTooltip,
  ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine,
} from 'recharts';

import faceRaw       from '../../data/face_emtions.json';
import audioRaw      from '../../data/audio_features.json';
import textRaw       from '../../data/text_emotions.json';
import transcriptRaw from '../../data/transcript.json';

// ─── Types ────────────────────────────────────────────────────────────────────

interface FaceEntry {
  anger_idx: number; arousal: number; comtempt_idx: number;
  disgust_idx: number; fear_idx: number; happy_idx: number;
  neutral_idx: number; sad_idx: number; surprise_idx: number;
  timestamp: number; valence: number; emotion: string;
}
interface AudioEntry {
  audio_emotions: { neu: number; hap: number; ang: number; sad: number };
  audio_acoustics: { rms_energy: number; spectral_centroid: number };
}
interface TextEntry  { [emotion: string]: number }
interface Segment    { start: number; end: number; text: string }
interface Transcript { segments: Segment[] }

const faceData   = faceRaw       as unknown as Record<string, FaceEntry>;
const audioData  = audioRaw      as unknown as Record<string, AudioEntry>;
const textData   = textRaw       as unknown as Record<string, TextEntry>;
const transcript = transcriptRaw as unknown as Transcript;

// ─── Constants ────────────────────────────────────────────────────────────────

const MAX_RMS = 0.009;
const WIN     = 10; // face history window in seconds

const FACE_KEYS = [
  'anger_idx','fear_idx','happy_idx',
  'neutral_idx','sad_idx','surprise_idx',
] as const;

type FaceKey = typeof FACE_KEYS[number];

const FACE_LABEL: Record<FaceKey, string> = {
  anger_idx:   'Anger',
  fear_idx:    'Fear',
  happy_idx:   'Happy',
  neutral_idx: 'Neutral',
  sad_idx:     'Sad',
  surprise_idx:'Surprise',
};

const C: Record<string, string> = {
  anger_idx:   '#f87171',
  fear_idx:    '#c084fc',
  happy_idx:   '#fde047',
  neutral_idx: '#94a3b8',
  sad_idx:     '#60a5fa',
  surprise_idx:'#fb923c',
  neu:'#94a3b8', hap:'#fde047', ang:'#f87171', sad:'#60a5fa',
};

const AUDIO_LABEL: Record<string,string> = {
  neu:'Neutral', hap:'Happy', ang:'Angry', sad:'Sad',
};

// RMS below this = treat as silence for "stale" purposes
const RMS_SILENCE = 0.002;

// ─── usePersisted ─────────────────────────────────────────────────────────────
// Always returns last non-empty value. stale=true when current is absent/empty.
function usePersisted<T>(
  current: T | null,
  isEmpty: (v: T) => boolean
): { value: T | null; stale: boolean } {
  const lastRef = useRef<T | null>(null);
  if (current !== null && !isEmpty(current)) lastRef.current = current;
  return { value: lastRef.current, stale: current === null || isEmpty(current) };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function topN(obj: Record<string,number>, n: number): [string,number][] {
  return Object.entries(obj).sort(([,a],[,b]) => b-a).slice(0,n) as [string,number][];
}

interface HistPt { sec: number; [k: string]: number }

// ─── Sub-components ───────────────────────────────────────────────────────────

function Panel({ style, children }: { style: React.CSSProperties; children: React.ReactNode }) {
  return (
    <div style={{
      position: 'absolute',
      borderRadius: 14,
      padding: '12px 14px',
      background: 'rgba(4,4,14,0.62)',
      backdropFilter: 'blur(20px) saturate(1.4)',
      WebkitBackdropFilter: 'blur(20px) saturate(1.4)',
      border: '1px solid rgba(255,255,255,0.08)',
      boxShadow: '0 8px 40px rgba(0,0,0,0.65), inset 0 1px 0 rgba(255,255,255,0.05)',
      color: '#fff',
      fontFamily: "'IBM Plex Mono', monospace",
      overflow: 'hidden',
      ...style,
    }}>
      {children}
    </div>
  );
}

function PanelLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 8, letterSpacing: '0.22em', textTransform: 'uppercase',
      color: 'rgba(255,255,255,0.28)',
      display: 'flex', alignItems: 'center', gap: 6,
    }}>
      <span style={{ display:'inline-block', width:14, height:1, background:'rgba(255,255,255,0.2)' }}/>
      {children}
    </div>
  );
}

function StalePill() {
  return (
    <span style={{
      fontSize: 7, letterSpacing: '0.12em', textTransform: 'uppercase',
      color: 'rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.05)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 99, padding: '1px 6px', marginLeft: 8, whiteSpace: 'nowrap',
    }}>
      last known
    </span>
  );
}

function Chip({ label, pct, color }: { label:string; pct:number; color:string }) {
  return (
    <span style={{
      display:'inline-flex', alignItems:'center', gap:5,
      padding:'3px 9px', borderRadius:999, fontSize:9,
      background:color+'22', border:`1px solid ${color}55`, color,
    }}>
      <span style={{ width:5, height:5, borderRadius:'50%', background:color, flexShrink:0 }}/>
      {label} <span style={{opacity:.6}}>{pct.toFixed(0)}%</span>
    </span>
  );
}

function HudTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background:'rgba(4,4,14,0.93)', border:'1px solid rgba(255,255,255,0.1)',
      borderRadius:8, padding:'6px 10px', fontSize:9,
      fontFamily:"'IBM Plex Mono',monospace",
    }}>
      {payload.map((p: any) => (
        <div key={p.dataKey} style={{ color:p.stroke??p.fill, display:'flex', gap:10, justifyContent:'space-between' }}>
          <span>{FACE_LABEL[p.dataKey as FaceKey] ?? AUDIO_LABEL[p.dataKey] ?? p.dataKey}</span>
          <span>{(+p.value).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const brainVideoRef = useRef<HTMLVideoElement>(null);
  const [sec, setSec]   = useState(0);
  const [time, setTime] = useState(0);
  const [rightTab, setRightTab] = useState<'audio' | 'brain'>('audio');
  const histRef         = useRef<HistPt[]>([]);
  const audioHistRef    = useRef<Array<{sec: number; [k: string]: number}>>([]);
  const [history, setHistory] = useState<HistPt[]>([]);
  const [audioHist, setAudioHist] = useState<Array<{sec: number; [k: string]: number}>>([]);

  const [facialVisible, setFacialVisible] = useState(true);
  const [rightVisible, setRightVisible] = useState(true);
  const [transcriptVisible, setTranscriptVisible] = useState(true);

  // rAF loop
  useEffect(() => {
    let raf: number;
    const tick = () => {
      if (videoRef.current && brainVideoRef.current) {
        const t = videoRef.current.currentTime;
        setTime(t);
        setSec(prev => { const n = Math.floor(t); return n !== prev ? n : prev; });
        // Sync brain video: update frequently for smooth playback
        if (Math.abs(brainVideoRef.current.currentTime - t) > 0.05) {
          brainVideoRef.current.currentTime = t;
        }
        // Sync play/pause state
        if (videoRef.current.paused && !brainVideoRef.current.paused) {
          brainVideoRef.current.pause();
        } else if (!videoRef.current.paused && brainVideoRef.current.paused) {
          brainVideoRef.current.play().catch(() => {});
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Build rolling face history
  useEffect(() => {
    const f = faceData[sec.toString()];
    if (!f) return;
    const pt: HistPt = { sec };
    FACE_KEYS.forEach(k => { pt[k] = +((f[k as keyof FaceEntry] as number) * 100).toFixed(2); });
    const next = [...histRef.current.filter(p => p.sec >= sec - WIN), pt];
    histRef.current = next;
    setHistory([...next]);
  }, [sec]);

  // Build rolling audio history (3-second window)
  useEffect(() => {
    const a = audioData[sec.toString()];
    if (!a) return;
    const AUDIO_WIN = 3;
    const pt: {sec: number; [k: string]: number} = { sec };
    Object.entries(a.audio_emotions).forEach(([k, v]) => {
      pt[k] = +(v * 100).toFixed(1);
    });
    const next = [...audioHistRef.current.filter(p => p.sec >= sec - AUDIO_WIN), pt];
    audioHistRef.current = next;
    setAudioHist([...next]);
  }, [sec]);

  // ── Raw current data ────────────────────────────────────────────────────────
  const face  = faceData[sec.toString()]  ?? null;
  const audio = audioData[sec.toString()] ?? null;
  const textCurrent = textData[sec.toString()] ?? {};
  const currentSeg  = transcript.segments.find(s => time >= s.start && time < s.end) ?? null;

  // ── Persisted (never blank) ─────────────────────────────────────────────────
  const { value: pAudio, stale: audioStale } = usePersisted(
    audio, a => a.audio_acoustics.rms_energy < RMS_SILENCE
  );
  const { value: pText, stale: textStale } = usePersisted(
    Object.keys(textCurrent).length > 0 ? textCurrent : null,
    t => Object.keys(t).length === 0
  );
  const { value: pSeg, stale: subStale } = usePersisted(currentSeg, () => false);

  const dAudio = pAudio ?? audio;
  const dText  = pText  ?? {};

  // ── Derived ─────────────────────────────────────────────────────────────────
  const topFaceKeys: FaceKey[] = face
    ? topN(Object.fromEntries(FACE_KEYS.map(k => [k, face[k as keyof FaceEntry] as number])), 3).map(([k]) => k) as FaceKey[]
    : ['fear_idx','neutral_idx','sad_idx'] as FaceKey[];

  // Compute averaged audio bars from history
  const audioBars = audioHist.length > 0
    ? Object.keys(audioHist[0])
        .filter(k => k !== 'sec')
        .map(k => {
          const avg = audioHist.reduce((sum, pt) => sum + (pt[k] ?? 0), 0) / audioHist.length;
          return { k, label: AUDIO_LABEL[k] ?? k, v: +avg.toFixed(1) };
        })
    : [];

  const rmsNorm  = dAudio ? Math.min((dAudio.audio_acoustics.rms_energy / MAX_RMS)*100, 100) : 0;
  const domColor = face ? (C[`${face.emotion.toLowerCase()}_idx`] ?? '#fff') : '#fff';

  return (
    <div style={{ position:'fixed', inset:0, width:'100vw', height:'100vh', overflow:'hidden', background:'#000' }}>

      {/* VIDEO — full bleed */}
      <video
        ref={videoRef}
        src="src/assets/clip.mp4"
        controls
        style={{ position:'absolute', inset:0, width:'100%', height:'100%', objectFit:'cover', zIndex:0 }}
      />

      {/* Vignette */}
      <div style={{
        position:'absolute', inset:0, zIndex:1, pointerEvents:'none',
        background:'radial-gradient(ellipse at 50% 50%, transparent 35%, rgba(0,0,0,0.6) 100%)',
      }}/>

      {/* ── HEADER ────────────────────────────────────────────────────────── */}
      <Panel style={{
        top:14, left:'50%', transform:'translateX(-50%)',
        display:'flex', alignItems:'center', gap:20,
        padding:'8px 22px', zIndex:20, whiteSpace:'nowrap',
      }}>
        <span style={{ fontSize:9, fontWeight:300, letterSpacing:'0.28em', color:'rgba(255,255,255,0.55)', textTransform:'uppercase' }}>
          In-Silico Neuroscience
        </span>
        <span style={{ fontSize:12, fontWeight:600, letterSpacing:'0.2em' }}>TRIBEv2</span>
        <span style={{ fontSize:9, color:'rgba(255,255,255,0.3)', fontVariantNumeric:'tabular-nums' }}>
          T+{time.toFixed(1)}s
        </span>
        {face && (
          <span style={{ fontSize:9, color:domColor, fontVariantNumeric:'tabular-nums' }}>
            ● {face.emotion.toUpperCase()}
          </span>
        )}
        <div style={{display:'flex', gap:5, marginLeft:20}}>
          <button style={{fontSize:8, background:'rgba(255,255,255,0.08)', border:'1px solid rgba(255,255,255,0.2)', color:'white', padding:'2px 8px', cursor:'pointer', borderRadius:999}} onClick={() => setFacialVisible(!facialVisible)}>{facialVisible ? 'Hide Facial' : 'Show Facial'}</button>
          <button style={{fontSize:8, background:'rgba(255,255,255,0.08)', border:'1px solid rgba(255,255,255,0.2)', color:'white', padding:'2px 8px', cursor:'pointer', borderRadius:999}} onClick={() => setRightVisible(!rightVisible)}>{rightVisible ? 'Hide Right' : 'Show Right'}</button>
          <button style={{fontSize:8, background:'rgba(255,255,255,0.08)', border:'1px solid rgba(255,255,255,0.2)', color:'white', padding:'2px 8px', cursor:'pointer', borderRadius:999}} onClick={() => setTranscriptVisible(!transcriptVisible)}>{transcriptVisible ? 'Hide Transcript' : 'Show Transcript'}</button>
        </div>
      </Panel>

      {/* ── TOP-LEFT: Facial ECG ─────────────────────────────────────── */}
      {facialVisible && (
      <Panel style={{ top:68, left:16, width:320, height:280, zIndex:20, display:'flex', flexDirection:'column', resize:'both', overflow:'hidden' }}>
      <button style={{position:'absolute', top:5, right:5, background:'none', border:'none', color:'white', fontSize:12, cursor:'pointer', zIndex:10}} onClick={() => setFacialVisible(false)}>×</button>

        <PanelLabel>Facial Micro-Expressions</PanelLabel>

        {face && (
          <div style={{ display:'flex', gap:16, fontSize:9, color:'rgba(255,255,255,0.35)',
                        marginBottom:6, fontVariantNumeric:'tabular-nums' }}>
            <span>V <b style={{color:'rgba(255,255,255,0.75)'}}>{face.valence>=0?'+':''}{face.valence.toFixed(3)}</b></span>
            <span>A <b style={{color:'rgba(255,255,255,0.75)'}}>{face.arousal.toFixed(3)}</b></span>
          </div>
        )}

        {/* Line chart */}
        <div style={{ flex:1, minHeight:0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top:4, right:4, bottom:0, left:-28 }}>
              <XAxis dataKey="sec" hide />
              <YAxis domain={[0,100]} tick={{ fill:'rgba(255,255,255,0.2)', fontSize:8 }}
                     tickLine={false} axisLine={false} tickFormatter={v=>`${v}%`}/>
              <ReTooltip content={<HudTooltip />}/>
              {topFaceKeys.map(k => (
                <Line key={k} type="natural" dataKey={k}
                      stroke={C[k]} strokeWidth={1.5}
                      dot={false} isAnimationActive={false}/>
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ── LEGEND ── */}
        <div style={{
          display:'flex', flexWrap:'wrap', gap:'5px 14px',
          marginTop:6, paddingTop:6,
          borderTop:'1px solid rgba(255,255,255,0.07)',
        }}>
          {topFaceKeys.map(k => (
            <span key={k} style={{
              display:'inline-flex', alignItems:'center', gap:5,
              fontSize:8, color:C[k], letterSpacing:'0.06em',
            }}>
              {/* line swatch */}
              <svg width="16" height="6" style={{ flexShrink:0 }}>
                <line x1="0" y1="3" x2="16" y2="3" stroke={C[k]} strokeWidth="2"/>
              </svg>
              {FACE_LABEL[k]}
            </span>
          ))}
        </div>
      </Panel>
      )}

      {/* ── BOTTOM-CENTER: Transcript + Text Sentiment ─────────────────────── */}
      {transcriptVisible && (
      <Panel style={{ bottom:24, left:'50%', transform:'translateX(-50%)', width:320, zIndex:20, display:'flex', flexDirection:'column' }}>
      <button style={{position:'absolute', top:5, right:5, background:'none', border:'none', color:'white', fontSize:12, cursor:'pointer'}} onClick={() => setTranscriptVisible(false)}>×</button>

        {/* Header row */}
        <div style={{ display:'flex', alignItems:'center', marginBottom:8 }}>
          <PanelLabel>Semantic Processing</PanelLabel>
          {textStale && <StalePill />}
        </div>

        {/* Subtitle — dims when stale, never blanks */}
        <p style={{
          fontSize:15, fontStyle:'italic', lineHeight:1.4,
          color: subStale ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.88)',
          minHeight:36, marginBottom:10,
          fontFamily:'Georgia, serif',
          transition:'color 0.8s ease',
        }}>
          {pSeg
            ? `"${pSeg.text.trim()}"`
            : <span style={{fontStyle:'normal',fontSize:11,color:'rgba(255,255,255,0.12)'}}>—</span>
          }
        </p>

        {/* Emotion chips */}
        <div style={{ display:'flex', flexWrap:'wrap', gap:5,
                      opacity: textStale ? 0.38 : 1, transition:'opacity 0.8s ease' }}>
          {topN(dText, 3).map(([label,val]) => (
            <Chip key={label} label={label} pct={val*100} color="#60a5fa"/>
          ))}
        </div>
      </Panel>
      )}

      {/* ── RIGHT: Audio + Brain (tabs) ───────────────────────────────────── */}
      {rightVisible && (
      <Panel style={{ top:68, right:16, width:288, height:320, zIndex:20, display:'flex', flexDirection:'column', resize:'both', overflow:'hidden', minWidth:240, minHeight:240 }}>
      <button style={{position:'absolute', top:5, right:5, background:'none', border:'none', color:'white', fontSize:12, cursor:'pointer', zIndex:10}} onClick={() => setRightVisible(false)}>×</button>

        {/* Tab buttons */}
        <div style={{ display:'flex', gap:8, marginBottom:12, borderBottom:'1px solid rgba(255,255,255,0.07)', paddingBottom:8 }}>
          <button
            onClick={() => setRightTab('audio')}
            style={{
              background:'none', border:'none', color: rightTab === 'audio' ? '#fff' : 'rgba(255,255,255,0.4)',
              fontSize:9, fontWeight: rightTab === 'audio' ? 600 : 400, letterSpacing:'0.1em', cursor:'pointer',
              padding:'2px 8px', borderBottom: rightTab === 'audio' ? '2px solid #60a5fa' : 'none',
              transition:'all 0.2s ease', fontFamily:"'IBM Plex Mono', monospace",
            }}
          >
            AUDIO
          </button>
          <button
            onClick={() => setRightTab('brain')}
            style={{
              background:'none', border:'none', color: rightTab === 'brain' ? '#fff' : 'rgba(255,255,255,0.4)',
              fontSize:9, fontWeight: rightTab === 'brain' ? 600 : 400, letterSpacing:'0.1em', cursor:'pointer',
              padding:'2px 8px', borderBottom: rightTab === 'brain' ? '2px solid #818cf8' : 'none',
              transition:'all 0.2s ease', fontFamily:"'IBM Plex Mono', monospace",
            }}
          >
            BRAIN
          </button>
        </div>

        {/* Audio tab */}
        {rightTab === 'audio' && (
          <>
            {/* Audio header */}
            <div style={{ display:'flex', alignItems:'center', marginBottom:8 }}>
              <PanelLabel>Acoustic &amp; Vocal Sentiment</PanelLabel>
              {audioStale && <StalePill />}
            </div>

            {/* RMS bar */}
            <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:10,
                          fontSize:8, color:'rgba(255,255,255,0.28)',
                          opacity: audioStale ? 0.38 : 1, transition:'opacity 0.8s ease' }}>
              <span>RMS</span>
              <div style={{ flex:1, height:3, borderRadius:3, background:'rgba(255,255,255,0.08)', overflow:'hidden' }}>
                <div style={{
                  height:'100%', borderRadius:3,
                  background:'linear-gradient(90deg,#fde047,#f97316)',
                  width:`${rmsNorm}%`, transition:'width 0.2s ease',
                }}/>
              </div>
              <span style={{ fontVariantNumeric:'tabular-nums', width:48, textAlign:'right' }}>
                {dAudio ? dAudio.audio_acoustics.rms_energy.toExponential(2) : '—'}
              </span>
            </div>

            {/* BarChart */}
            <div style={{ flex:'1 1 0', minHeight:0, opacity: audioStale ? 0.38 : 1, transition:'opacity 0.8s ease' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={audioBars} margin={{ top:4, right:4, bottom:0, left:-24 }} barCategoryGap="22%">
                  <XAxis dataKey="label" tick={{ fill:'rgba(255,255,255,0.45)', fontSize:9 }}
                         tickLine={false} axisLine={false}/>
                  <YAxis domain={[0,100]} tick={{ fill:'rgba(255,255,255,0.2)', fontSize:8 }}
                         tickLine={false} axisLine={false} tickFormatter={v=>`${v}%`}/>
                  <ReTooltip content={<HudTooltip />}/>
                  <ReferenceLine y={rmsNorm} stroke="rgba(253,224,71,0.4)"
                                 strokeDasharray="3 3" strokeWidth={1}/>
                  <Bar dataKey="v" radius={[3,3,0,0]} isAnimationActive={false}>
                    {audioBars.map(d => (
                      <Cell key={d.k} fill={C[d.k] ?? '#6366f1'}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* Brain tab */}
        {rightTab === 'brain' && (
          <div style={{ display:'flex', flexDirection:'column', gap:6, flex:1, minHeight:0 }}>
            <PanelLabel>TRIBEv2 Neural Activations</PanelLabel>
            <video
              ref={brainVideoRef}
              src="src/assets/brain_responses_video.mp4"
              style={{ flex:1, width:'100%', height:'100%', borderRadius:6, backgroundColor:'#000', objectFit:'contain' }}
              muted
            />
          </div>
        )}

        {/* Keep brain video in DOM when audio tab active (hidden but syncing) */}
        {rightTab === 'audio' && (
          <video
            ref={brainVideoRef}
            src="src/assets/brain_responses_video.mp4"
            style={{ position:'fixed', left:'-9999px', pointerEvents:'none' }}
            muted
          />
        )}
      </Panel>
      )}
      <style>{`
        @keyframes pulse  { 0%,100%{opacity:.22} 50%{opacity:.38} }
        @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&display=swap');
      `}</style>
    </div>
  );
}