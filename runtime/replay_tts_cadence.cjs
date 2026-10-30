const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const crypto=require('node:crypto');
const root=path.resolve(__dirname,'..');
const playback=require(path.join(root,'portal/static/call_playback.js'));
const source=fs.readFileSync(path.join(root,'portal/static/portal.js'),'utf8');
const start=source.indexOf('  function pendingSeconds(');
const end=source.indexOf('  async function unlockPlayback(',start);
if(start<0||end<start)throw Error('source boundaries not found');
const constants=[...source.matchAll(/^\s*const (PCM_[A-Z_]+) = ([\d.]+);/gm)].map(m=>`const ${m[1]}=${m[2]};`).join('\n');
function replay(run,priorRate){
 const intervals=[];
 const context={currentTime:0,destination:{},createBuffer:(channels,samples,rate)=>({duration:samples/rate,copyToChannel:()=>{}}),createGain:()=>({connect(){},gain:{setValueAtTime(){},linearRampToValueAtTime(){}}}),createBufferSource:()=>({connect(){},addEventListener(){},start(t){intervals.push({start:t,end:t+this.buffer.duration});}})};
 const sandbox={callPlayback:playback,state:{playbackEpoch:1},atob:v=>Buffer.from(v,'base64').toString('binary'),Float32Array,Set,maybeFinishPcmPlayback:()=>{}};
 vm.createContext(sandbox);vm.runInContext(constants+'\n'+source.slice(start,end)+'\nglobalThis.queue=queuePcmPlayback;globalThis.finish=endPcmPlayback;',sandbox);
 const controller={epoch:1,context,pacer:playback.createPacer({prior:{rate:priorRate,charsPerSecond:14},textChars:run.text_chars}),pending:[],started:false,nextTime:0,lastGain:null,lastDuration:0,sources:new Set(),ended:false,cancelled:false};
 for(const arrival of run.arrivals){context.currentTime=arrival.time_s;sandbox.queue(controller,Buffer.alloc(arrival.bytes).toString('base64'));}
 context.currentTime=run.total_s;sandbox.finish(controller);
 return {prior_rate:priorRate,first_scheduled_audio_s:intervals[0].start,scheduled_completion_s:Math.max(...intervals.map(x=>x.end)),underruns:controller.pacer.underruns,required_lead_s:playback.requiredLead(controller.pacer)};
}
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
for(const c of data.cases)for(const r of c.runs)r.browser_replay=[.8,1,1.6].map(p=>replay(r,p));
data.playback_source_sha256=crypto.createHash('sha256').update(source).digest('hex');
data.playback_pacer_sha256=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'portal/static/call_playback.js'))).digest('hex');
data.replay_scope='Unmodified browser scheduling/pacer functions with recorded native arrivals, no network delay or physical speaker latency; zero buffers stand in for sample amplitudes only.';
fs.writeFileSync(process.argv[3],JSON.stringify(data,null,2)+'\n');
for(const c of data.cases){const runs=c.runs.slice(1);console.log(JSON.stringify({frames:c.frames,replays:runs.map(r=>r.browser_replay.map(x=>({prior:x.prior_rate,first:x.first_scheduled_audio_s,total:x.scheduled_completion_s,underruns:x.underruns})))}));}
