/* Display geometry, bitmap text, canopy and label dynamics from the supplied index(9).html.
   The feed is the authenticated Omni snapshot. */
(() => {
"use strict";


const VIEW = Object.freeze({ width:910, height:1630, x:58, top:212, right:576, bottom:1454, axisX:628 });
const PAL = Object.freeze({
  black:'#020101', paper:'#060201',
  hot:'#fff568', bright:'#ffd14a', primary:'#ffaa20', orange:'#f08a17',
  medium:'#aa4d0e', subdued:'#613010', faint:'#34170c', ghost:'#210f08',
  ink:'#170b03'
});

const CHANNELS = [
  ["input", "INPUT", 375], ["heard", "HEARD", 478], ["inference", "INFER", 616],
  ["tools", "TOOLS", 745], ["memory", "MEMORY", 875], ["media", "MEDIA", 1007],
  ["said", "SAID", 1130], ["tasks", "TASKS", 1260], ["voice", "VOICE", 1390],
];
const CHANNEL_Y = new Map(CHANNELS.map(([key, , y]) => [key, y]));
const CHANNEL_NAMES = new Map(CHANNELS.map(([key, label]) => [key, label]));


const TIME_MARKS = [
  { seconds:147, text:'2:27', x:70 },
  { seconds:60, text:'1m', x:245 },
  { seconds:15, text:'15s', x:366 },
  { seconds:5, text:'5s', x:426 },
  { seconds:0, text:'NOW', x:576 }
];

function ageX(age){
  const a=Math.max(0,Math.min(147,age));
  for(let i=0;i<TIME_MARKS.length-1;i++){
    const q=TIME_MARKS[i], n=TIME_MARKS[i+1];
    if(a<=q.seconds && a>=n.seconds){return n.x + (q.x-n.x)*(a-n.seconds)/(q.seconds-n.seconds);}
  }
  return a>=147?TIME_MARKS[0].x:TIME_MARKS[TIME_MARKS.length-1].x;
}

const FONT = {
  'A':['01110','10001','10001','11111','10001','10001','10001'],
  'B':['11110','10001','10001','11110','10001','10001','11110'],
  'C':['01111','10000','10000','10000','10000','10000','01111'],
  'D':['11110','10001','10001','10001','10001','10001','11110'],
  'E':['11111','10000','10000','11110','10000','10000','11111'],
  'F':['11111','10000','10000','11110','10000','10000','10000'],
  'G':['01111','10000','10000','10111','10001','10001','01111'],
  'H':['10001','10001','10001','11111','10001','10001','10001'],
  'I':['11111','00100','00100','00100','00100','00100','11111'],
  'J':['00111','00010','00010','00010','10010','10010','01100'],
  'K':['10001','10010','10100','11000','10100','10010','10001'],
  'L':['10000','10000','10000','10000','10000','10000','11111'],
  'M':['10001','11011','10101','10101','10001','10001','10001'],
  'N':['10001','11001','10101','10011','10001','10001','10001'],
  'O':['01110','10001','10001','10001','10001','10001','01110'],
  'P':['11110','10001','10001','11110','10000','10000','10000'],
  'Q':['01110','10001','10001','10001','10101','10010','01101'],
  'R':['11110','10001','10001','11110','10100','10010','10001'],
  'S':['01111','10000','10000','01110','00001','00001','11110'],
  'T':['11111','00100','00100','00100','00100','00100','00100'],
  'U':['10001','10001','10001','10001','10001','10001','01110'],
  'V':['10001','10001','10001','10001','10001','01010','00100'],
  'W':['10001','10001','10001','10101','10101','10101','01010'],
  'X':['10001','10001','01010','00100','01010','10001','10001'],
  'Y':['10001','10001','01010','00100','00100','00100','00100'],
  'Z':['11111','00001','00010','00100','01000','10000','11111'],
  '0':['01110','10011','10101','10101','10101','11001','01110'],
  '1':['00100','01100','00100','00100','00100','00100','01110'],
  '2':['01110','10001','00001','00010','00100','01000','11111'],
  '3':['11110','00001','00001','01110','00001','00001','11110'],
  '4':['00010','00110','01010','10010','11111','00010','00010'],
  '5':['11111','10000','10000','11110','00001','00001','11110'],
  '6':['01110','10000','10000','11110','10001','10001','01110'],
  '7':['11111','00001','00010','00100','01000','01000','01000'],
  '8':['01110','10001','10001','01110','10001','10001','01110'],
  '9':['01110','10001','10001','01111','00001','00001','01110'],
  '.':['00000','00000','00000','00000','00000','01100','01100'],
  ':':['00000','01100','01100','00000','01100','01100','00000'],
  '-':['00000','00000','00000','11111','00000','00000','00000'],
  '/':['00001','00001','00010','00100','01000','10000','10000'],
  '+':['00000','00100','00100','11111','00100','00100','00000'],
  '%':['11001','11010','00100','00100','01011','10011','00000'],
  '>':['10000','01000','00100','00010','00100','01000','10000'],
  '<':['00001','00010','00100','01000','00100','00010','00001'],
  '=':['00000','11111','00000','11111','00000','00000','00000'],
  '_':['00000','00000','00000','00000','00000','00000','11111'],
  '[':['01110','01000','01000','01000','01000','01000','01110'],
  ']':['01110','00010','00010','00010','00010','00010','01110'],
  '?':['11110','00001','00001','00110','00100','00000','00100'],
  '!':['00100','00100','00100','00100','00100','00000','00100'],
  ' ':['00000','00000','00000','00000','00000','00000','00000']
};
class Pixel {
  constructor(ctx){this.c=ctx;this.color=PAL.primary;this.glow=false;this.s=2.4;}
  text(str,x,y,size=2.4,color=PAL.primary,{glow=false,align='left'}={}){
    const c=this.c,chars=String(str).toUpperCase().replaceAll('—','-'),step=size*6;
    if(align==='right')x-=chars.length*step;
    if(align==='center')x-=(chars.length*step)/2;
    c.save();c.fillStyle=color;
    if(glow){c.shadowColor=color;c.shadowBlur=Math.min(8,size*2.8)}
    for(let i=0;i<chars.length;i++){
      const glyph=FONT[chars[i]]||FONT['?'];
      for(let r=0;r<7;r++)for(let col=0;col<5;col++){
        if(glyph[r].charCodeAt(col)===49)c.fillRect(Math.round(x+i*step+col*size),Math.round(y+r*size),Math.ceil(size+.09),Math.ceil(size+.09));
      }
    }c.restore();return chars.length*step;
  }
  rect(x,y,w,h,col=PAL.primary,alpha=1){const c=this.c;c.save();c.globalAlpha=alpha;c.fillStyle=col;c.fillRect(Math.round(x),Math.round(y),Math.max(0,Math.round(w)),Math.max(0,Math.round(h)));c.restore()}
  line(x1,y1,x2,y2,color=PAL.medium,width=2,alpha=1){const c=this.c;c.save();c.globalAlpha=alpha;c.strokeStyle=color;c.lineWidth=width;c.beginPath();c.moveTo(Math.round(x1),Math.round(y1));c.lineTo(Math.round(x2),Math.round(y2));c.stroke();c.restore()}
  poly(points,color=PAL.primary,width=2,alpha=1){if(!points.length)return;const c=this.c;c.save();c.globalAlpha=alpha;c.strokeStyle=color;c.lineWidth=width;c.lineJoin='miter';c.beginPath();points.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.stroke();c.restore()}
  dot(x,y,r=4,color=PAL.bright,glow=true){const c=this.c;c.save();c.fillStyle=color;if(glow){c.shadowColor=color;c.shadowBlur=9}c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill();c.restore()}
  hatch(x,y,w,h,color=PAL.primary,dim=false){
    const c=this.c;c.save();c.beginPath();c.rect(x,y,w,h);c.clip();c.fillStyle=dim?'#1d0c03':'#4b1c03';c.fillRect(x,y,w,h);
    c.strokeStyle=dim?PAL.medium:color;c.lineWidth=4;
    for(let i=-h;i<w+h;i+=15){c.beginPath();c.moveTo(x+i,y+h);c.lineTo(x+i+h,y);c.stroke()}
    c.restore();this.line(x,y,x+w,y,color,2);this.line(x,y+h,x+w,y+h,color,2)
  }
  bracket(x,y,w,h,color=PAL.bright){
    this.poly([[x,y+11],[x,y],[x+w,y],[x+w,y+11]],color,4);
    this.poly([[x,y+h-11],[x,y+h],[x+w,y+h],[x+w,y+h-11]],color,4);
  }
}

const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
function hash01(seed,salt=0){
  let n=(Math.imul(seed|0,0x9e3779b1)^Math.imul(salt|0,0x85ebca6b))|0;
  n=Math.imul(n^(n>>>16),0x7feb352d);
  n=Math.imul(n^(n>>>15),0x846ca68b);
  return ((n^(n>>>16))>>>0)/4294967296;
}
class CanopyField{
  constructor(clearance=20){this.clearance=clamp(Number(clearance)||20,0,45);this.crowns=[];this.segments=[];}
  // The occupation circle represents the terminal crown, not the shared trunk.
  // Conservatively shrink to preserve a hard gap between distinct branch clusters.
  claim(x,y,direction,desiredHeight,eventId){
    let height=Math.max(0,desiredHeight),cx=x,cy=y,r=0;
    for(let pass=0;pass<12;pass++){
      cy=y+direction*height*.69;r=height*.44;
      let safe=true;
      for(const other of this.crowns){
        const dist=Math.hypot(cx-other.x,cy-other.y);
        const missing=other.r+r+this.clearance-dist;
        if(missing>0){height=Math.max(0,height-Math.max(1.25,missing*.9));safe=false;break;}
      }
      if(safe)break;
    }
    cy=y+direction*height*.69;r=height*.44;
    if(height<8)return null;
    // Final validation also protects against numeric tolerance near contacts.
    if(this.crowns.some(o=>Math.hypot(x-o.x,cy-o.y)<o.r+r+this.clearance-.1))return null;
    const crown={x,y:cy,r,eventId,height,direction};this.crowns.push(crown);return crown;
  }
  draw(p,e,x0,x1,y,palette,growth=1){
    const width=x1-x0;
    if(width<24)return;
    const n=clamp(Math.floor(width/62),1,7);
    for(let i=0;i<n;i++){
      const jitter=(hash01(e.id,i+40)-.5)*Math.min(12,width/n*.23);
      const x=clamp(x0+(i+.55)*width/n+jitter,70,569);
      const direction=hash01(e.id,i+91)<.23?1:-1;
      const height=Math.min(48,19+e.sig*.48+hash01(e.id,i+17)*12)*growth;
      // Stay inside the display, reserve breathing room at the channel axis.
      const room=direction<0?y-355:1441-y;
      const bounded=Math.min(height,room,Math.max(0,x-68)*1.15,Math.max(0,572-x)*1.15);
      const crown=this.claim(x,y,direction,bounded,e.id);
      if(!crown)continue;
      const h=crown.height,dir=direction;
      const lean=(hash01(e.id,i+313)-.5)*h*.32;
      const tipX=x+lean,tipY=y+dir*h;
      const forkX=x+lean*.46,forkY=y+dir*h*.57;
      const br=h*.32;
      const count=Math.max(1,Math.floor(h/11));
      // Pixel-circuit silhouettes: a central stem, two asymmetric fork tips,
      // and a tiny terminal blink. All tips remain inside the shy crown.
      p.line(x,y,forkX,forkY,palette.primary,2.6,.92);
      p.line(forkX,forkY,tipX,tipY,palette.bright,2.5,.88);
      if(count>1){
        p.poly([[forkX,forkY],[forkX-br*.66,forkY+dir*h*.14],[forkX-br*.89,forkY+dir*h*.31]],palette.primary,2.2,.74);
        p.poly([[forkX,forkY],[forkX+br*.62,forkY+dir*h*.16],[forkX+br*.87,forkY+dir*h*.29]],palette.orange,2,.72);
      }
      // Secondary limblets make the clustered tips read as branching canopies,
      // while staying within the pre-cleared terminal circle.
      if(count>2){
        const ux=tipX-h*.13, vx=tipX+h*.13;
        const sy=tipY-dir*h*.19;
        p.line(tipX,tipY,ux,sy,palette.orange,1.8,.67);
        p.line(tipX,tipY,vx,sy,palette.primary,1.8,.72);
      }
      p.dot(tipX,tipY,2.1,palette.bright,false);
      this.segments.push({eventId:e.id,x,y,tipX,tipY,height:h});
    }
  }
}

// Independent SIG panel lifecycle and negative-space layout.
// A completed event does NOT leave a permanent box on the scope.
const clampLabel=(v,a,b)=>Math.max(a,Math.min(b,v));
const smooth=t=>{t=clampLabel(t,0,1);return t*t*(3-2*t)};
function annotationLife(event,time,hold=2.5){
  const sinceStart=time-event.start;
  if(sinceStart<.24)return null;
  const elapsed=event.end===null ? -1 : Math.max(0,time-event.end);
  const collapseAt=clampLabel(Number(hold)||2.5,.5,12);
  const collapseEnd=collapseAt+2.4;
  const retireAt=collapseEnd+5.2;
  const disappearAt=retireAt+2.1;
  if(elapsed>=disappearAt)return null;
  let expansion=1,alpha=1,stage='expanded';
  if(elapsed>=0){
    if(elapsed<collapseAt){stage='expanded';}
    else if(elapsed<collapseEnd){stage='collapsing';expansion=1-smooth((elapsed-collapseAt)/(collapseEnd-collapseAt));}
    else if(elapsed<retireAt){stage='compact';expansion=0;}
    else{stage='retiring';expansion=0;alpha=1-smooth((elapsed-retireAt)/(disappearAt-retireAt));}
  }
  const birth=smooth(sinceStart/.65);
  return {stage,expansion:expansion*birth,alpha:alpha*birth,secondsSinceEnd:elapsed};
}
function separated(a,b,gap){return a.x>=b.x+b.w+gap||a.x+a.w+gap<=b.x||a.y>=b.y+b.h+gap||a.y+a.h+gap<=b.y;}
// Layout depends only on inputs; deterministic permutations avoid frame-to-frame jitter.
function placeAnnotation(anchorX,anchorY,expansion,occupied,options={}){
  const w=62+66*expansion,h=25+62*expansion;
  const gap=options.gap??20;
  const compact=expansion<.45;
  const offsets=compact?[
    [12,-40],[-w-15,-40],[15,16],[-w-15,16],[55,-40],[-w-54,-40]
  ]:[
    [-w*.57,-h-35],[-w-16,30],[18,-h-31],[-w-16,-h-63],
    [-w-42,35],[27,30],[-w*.6,-h-100]
  ];
  let best=null,bestCost=Infinity;
  for(const [dx,dy] of offsets){
    const r={x:Math.round(clampLabel(anchorX+dx,80,566-w-7)),y:Math.round(clampLabel(anchorY+dy,359,1444-h-3)),w,h};
    if(occupied.some(o=>!separated(r,o,gap)))continue;
    // Prefer close labels with no excursion across the frequency axis.
    const cost=Math.abs((r.x+r.w*.5)-anchorX)*.25+Math.abs((r.y+r.h*.5)-anchorY)+
      (r.y>anchorY?8:0);
    if(cost<bestCost){best=r;bestCost=cost;}
  }
  return best;
}
function selectAnnotations(events,time,options={}){
  const hold=options.hold??2.5;
  const maxExpanded=options.maxExpanded??2;
  const maxCompact=options.maxCompact??2;
  const annotated=events.map(e=>({event:e,life:annotationLife(e,time,hold)}))
    .filter(a=>a.life && a.life.alpha>.03 && (a.event.sig>=20 || a.event.kind==='hatched' || a.event.sig>=13&&a.event.dur>6));
  annotated.sort((a,b)=>{
    const pa=(a.event.end===null?170:0)+a.event.sig*1.7+(a.life.stage==='expanded'?65:0)+a.event.start*.025;
    const pb=(b.event.end===null?170:0)+b.event.sig*1.7+(b.life.stage==='expanded'?65:0)+b.event.start*.025;
    return pb-pa || b.event.id-a.event.id;
  });
  let expanded=0,compact=0;
  const output=[];
  for(const row of annotated){
    const life={...row.life};
    if(life.expansion>.35){
      if(expanded>=maxExpanded){life.expansion=0;life.stage='compact';}
      else expanded++;
    }
    if(life.expansion<=.35){if(compact>=maxCompact)continue;compact++;}
    output.push({...row,life});
    if(expanded>=maxExpanded&&compact>=maxCompact)break;
  }
  return output;
}

const clip=(x,a,b)=>Math.max(a,Math.min(x,b));
class Renderer {
  constructor(canvas,sim,settings){this.canvas=canvas;this.c=canvas.getContext('2d',{alpha:false});this.p=new Pixel(this.c);this.sim=sim;this.settings=settings;this.hits=[];this.hover=null;this.flash=0;this.noise=this.makeNoise();this.allEvents=[];this.selected=null}
  makeNoise(){const c=document.createElement('canvas');c.width=512;c.height=512;const cx=c.getContext('2d');const im=cx.createImageData(512,512);let n=0x41ad203;for(let i=0;i<im.data.length;i+=4){n^=n<<13;n^=n>>>17;n^=n<<5;const v=(n>>>0)%100;im.data[i]=220;im.data[i+1]=110;im.data[i+2]=30;im.data[i+3]=v<4?15:0;}cx.putImageData(im,0,0);return c}
  render(){
    const c=this.c,p=this.p,s=this.settings;this.hits.length=0;
    c.fillStyle=PAL.black;c.fillRect(0,0,VIEW.width,VIEW.height);
    this.drawAmbient();this.drawPanel();this.drawTimeline();this.drawGrid();this.drawChannels();
    this.drawEvents();this.drawCallouts();this.drawSettings();this.drawFooter();this.drawBottom();this.drawNoise();
    if(s.debug)this.drawHotspots();
  }
  drawAmbient(){const c=this.c;
    const radial=c.createRadialGradient(350,800,250,350,800,970);radial.addColorStop(0,'#0e0501');radial.addColorStop(.6,'#060301');radial.addColorStop(1,'#020100');c.fillStyle=radial;c.fillRect(0,0,910,1630);
    const linear=c.createLinearGradient(0,64,0,122);linear.addColorStop(0,'#080609');linear.addColorStop(.5,'#1b1920');linear.addColorStop(1,'#050407');c.globalAlpha=.56;c.fillStyle=linear;c.fillRect(0,38,910,94);c.globalAlpha=1;
  }
  drawPanel(){const p=this.p;
    p.line(VIEW.x,VIEW.top,VIEW.right,VIEW.top,PAL.medium,3,.88);
    p.line(VIEW.x,VIEW.top,VIEW.x,VIEW.bottom,PAL.medium,3,.83);
    p.line(VIEW.right,VIEW.top,VIEW.right,VIEW.bottom,PAL.orange,3,.90);
    p.line(VIEW.x,VIEW.bottom,VIEW.right,VIEW.bottom,PAL.orange,3,.95);
    // Fine retained CRT traces and separators.
    for(const x of [247,369,428])p.line(x,338,x,VIEW.bottom,PAL.faint,2,.72);
    for(const x of [91,101,103])p.line(x,365,x,VIEW.bottom,PAL.ghost,1,.28);
  }
  drawTimeline(){const p=this.p;
    p.line(75,VIEW.bottom+1,565,VIEW.bottom+1,PAL.primary,2);
    for(const mark of TIME_MARKS){if(mark.seconds===0)continue;
      const alpha=mark.seconds>=60?.27:.42;
      p.line(mark.x,1402,mark.x,1457,PAL.orange,3,alpha);
      p.rect(mark.x-3,1449,7,11,PAL.primary,.7);
    }
    // Timeline ticks along the bottom edge; a fading trace of activity.
    for(let i=0;i<30;i++){
      const x=82+i*15.3;
      if((i*7)%5!==0)p.rect(x,1450,2,5,i%3===0?PAL.medium:PAL.faint,.5);
    }
  }
  drawGrid(){
    const p=this.p;
    for(const [, ,y] of CHANNELS)p.line(576,y,617,y,PAL.medium,1,.33);
  }
  drawChannels(){
    const p=this.p;
    const events=this.allEvents.filter(event=>this.sim.t-event.end<=this.settings.window && this.sim.t>=event.end);
    for(const [key,label,y] of CHANNELS){
      const current=events.filter(event=>event.channel===key);
      const active=this.settings.channel===key||current.some(event=>this.sim.t-event.end<12);
      const x1=621,x2=637,x3=740,h=34;
      p.line(576,y,x1,y,active?PAL.orange:PAL.faint,2,active?.95:.75);
      p.poly([[x1,y],[x2,y-h/2],[x3,y-h/2],[x3,y+h/2],[x2,y+h/2],[x1,y]],
        active?PAL.bright:PAL.faint,active?3:2);
      if(active)p.rect(x2,y-h/2+2,x3-x2-2,h-4,PAL.hot,.98);
      p.text(label,x2+7,y-8,2.5,active?PAL.ink:PAL.medium);
      if(current.length)p.text(String(current.length),748,y-8,2.35,active?PAL.hot:PAL.medium);
      this.hits.push({x:621,y:y-20,w:151,h:41,type:'channel',value:key,label});
    }
  }
  drawEvents(){const c=this.c,p=this.p;
    c.save();c.beginPath();c.rect(60,345,514,1105);c.clip();
    const events=this.allEvents.filter(e=>this.sim.t-e.end<=this.settings.window && this.sim.t>=e.end && (this.settings.channel==='all'||e.channel===this.settings.channel));
    // Trace beds first, with crowns generated in a second pass.
    for(const e of events){
      const ageStart=this.sim.t-e.start,ageEnd=e.end===null?0:this.sim.t-e.end;
      const x0=clip(ageX(ageStart*147/this.settings.window),65,575),x1=clip(ageX(ageEnd*147/this.settings.window),65,575),y=CHANNEL_Y.get(e.channel);
      if(x1<x0||ageEnd>this.settings.window)continue;
      const k=e.kind,w=x1-x0,b=4+e.sig/8;
      const level=clip(e.sig+this.settings.gain,0,80);
      const gated=e.failed;
      const boosted=this.settings.compressor?Math.sqrt(level/45):level/45;
      const alpha=(ageStart>120?.60:1)*(gated?.35:clip(.64+boosted*.36,.4,1));
      c.globalAlpha=alpha;
      if(k==='hatched'){
        this.p.hatch(Math.max(66,x0),y-15,Math.min(575,x1)-Math.max(66,x0),35,PAL.bright,false);
        p.rect(x0,y-24,9,48,PAL.hot,.7);
      } else if(k==='pulse'){
        const px=x0+(x1-x0)*.5;
        p.line(px,y-4,px,y-30-b*1.5,PAL.primary,7,.9);
        p.rect(px-1,y-30-b*1.5,17,9,PAL.bright);
        p.line(px,y,px+12,y,PAL.medium,3);
        this.hits.push({x:px-10,y:y-36,w:22,h:43,type:'event',value:e.key,label:e.label});
      } else if(k==='pulses'){
        p.line(x0,y,x1,y,PAL.medium,4);
        let count=Math.max(3,Math.floor(w/17));for(let j=0;j<count;j++){
          let x=x0+j*(w/count)+Math.sin(j*7+e.phase)*8;
          p.line(x,y,x,y-9-Math.abs(Math.cos(j*11+e.phase))*Math.max(7,e.sig*.55),j%4===0?PAL.hot:PAL.medium,j%3===0?7:4);
          if(j%3===0)p.dot(x,y,4,PAL.bright,false);
        }
      } else {
        // Branch networks are laid out independently so crowns can avoid one another.
        p.line(x0,y,x1,y,PAL.primary,2.8,.85);
        if(w>25){
          p.dot(x0,y,3.5,PAL.orange,false);p.dot(x1,y,3.2,PAL.primary,false);
          const n=clip(Math.floor(w/85),1,6);
          for(let j=1;j<=n;j++){
            const xx=x0+w*j/(n+1);p.dot(xx,y,2.4,PAL.bright,false);
          }
        }
      }
      if(k!=='pulse')this.hits.push({x:x0-5,y:y-19,w:Math.max(14,x1-x0+10),h:38,type:'event',value:e.key,label:e.label});
      c.globalAlpha=1;
    }
    // Separate crown pass. New/live activity gets priority for clear growing tips.
    // Fixed decorative strokes were removed: all visible foliage is now event-driven.
    const forest=new CanopyField(this.settings.shyness);
    const ordered=events.filter(e=>e.kind==='trace' || e.kind==='pulses')
      .sort((a,b)=>(Number(b.end===null)-Number(a.end===null)) || b.sig-a.sig || b.start-a.start);
    for(const e of ordered){
      const startAge=this.sim.t-e.start,endAge=e.end===null?0:this.sim.t-e.end;
      const x0=clip(ageX(startAge*147/this.settings.window),65,575),x1=clip(ageX(endAge*147/this.settings.window),65,575);
      if(endAge>this.settings.window||x1-x0<24)continue;
      const y=CHANNEL_Y.get(e.channel);
      if(y<360||y>1435)continue;
      const growing=clip((this.sim.t-e.start-.15)/1.15,0,1);
      forest.draw(p,e,x0,x1,y,PAL,growing);
    }
    this.forest=forest;
    c.restore();
  }
  drawCallouts(){
    const occupied=[];
    const live=this.allEvents.filter(e=>this.sim.t>=e.end && this.sim.t-e.end<=this.settings.window
      && (this.settings.channel==='all'||e.channel===this.settings.channel));
    const rows=selectAnnotations(live,this.sim.t,{
      hold:this.settings.labelHold,maxExpanded:this.settings.maxCallouts,maxCompact:2
    });
    const selected=live.find(e=>e.key===this.selected);
    if(selected){
      const matching=rows.findIndex(row=>row.event.key===selected.key);
      if(matching>=0)rows.splice(matching,1);
      rows.unshift({event:selected,life:{alpha:1,expansion:1,stage:'selected'}});
    }
    for(const {event:e,life} of rows){

      const y=CHANNEL_Y.get(e.channel),ageEnd=e.end===null?0:this.sim.t-e.end;
      const anchorX=clip(ageX(ageEnd*147/this.settings.window),80,562);
      const rect=placeAnnotation(anchorX,y,life.expansion,occupied,{gap:this.settings.shyness});
      if(!rect)continue; // Explicitly suppress instead of piling boxes over one another.
      occupied.push(rect);
      this.drawCallout(e,rect,anchorX,y,life);
    }
    this.visibleAnnotations=occupied.length;
  }
  drawCallout(e,r,anchorX,anchorY,life){
    const p=this.p,c=this.c;
    const {x,y,w,h}=r,alpha=life.alpha,expanded=life.expansion;
    if(alpha<=.03)return;
    const connectorX=clip(anchorX,x+6,x+w-6);
    const connectorY=y+h<anchorY?y+h:y;
    p.poly([[connectorX,connectorY],[connectorX,connectorY+(connectorY<anchorY?9:-9)],
      [anchorX,anchorY]],PAL.primary,Math.max(1,1.5+expanded*.8),alpha*.67);
    p.dot(anchorX,anchorY,Math.max(1.5,2+expanded),PAL.bright,false);
    p.rect(x+2,y+2,w-4,h-4,PAL.black,.93*alpha);
    // Live SVG-like bracket morph: the full panel contracts into a single tiny tag.
    const k=Math.min(10,h*.27);
    p.poly([[x,y+k],[x,y],[x+w,y],[x+w,y+k]],PAL.bright,2+expanded*1.2,alpha);
    p.poly([[x,y+h-k],[x,y+h],[x+w,y+h],[x+w,y+h-k]],PAL.bright,2+expanded*1.2,alpha);
    c.save();c.beginPath();c.rect(x+3,y+2,w-6,h-4);c.clip();
    if(expanded>.4){
      const textScale=2.3+expanded*.6;
      p.text(CHANNEL_NAMES.get(e.channel),x+8,y+9,textScale,PAL.hot,{glow:expanded>.85});
      if(expanded>.76){
        p.text(e.short,x+8,y+36,2.05,PAL.primary);
        if(e.measure)p.text(e.measure,x+8,y+62,1.9,PAL.bright);
      }
    } else {
      p.text(CHANNEL_NAMES.get(e.channel),x+5,y+5,1.45,PAL.hot);
    }
    c.restore();
    // Click any visible SIG element to tune its source channel.
    this.hits.push({x,y,w,h,type:'event',value:e.key,label:e.label});
  }
  drawSettings(){
    const p=this.p;
    const keys=['all','heard','inference','tools','memory','media','said','tasks','voice'];
    keys.forEach((key,index)=>{
      const x=71+(index%3)*143,y=226+Math.floor(index/3)*40,w=130;
      const active=this.settings.channel===key;
      const value=key==='all'?'ALL':CHANNEL_NAMES.get(key);
      p.rect(x,y,w,30,active?PAL.primary:PAL.medium,.76);
      p.rect(x+3,y+3,w-6,24,active?PAL.hot:PAL.ink,.94);
      p.text(value,x+6,y+6,2.36,active?PAL.ink:PAL.bright,{glow:!active});
      p.line(x,y+28,x+w,y+28,PAL.primary,2,.75);
      this.hits.push({x,y:y-3,w,h:34,type:'channel',value:key,label:value});
    });
    const readout=this.sim.resources;
    if(Number.isFinite(readout.ram))p.text('RAM '+Math.round(readout.ram)+'%',589,229,2.45,PAL.medium);
    if(Number.isFinite(readout.loaded))p.text('WEIGHTS '+readout.loaded,589,265,2.2,PAL.medium);
  }
  drawFooter(){
    const p=this.p;
    for(const mark of TIME_MARKS.slice(0,-1)){
      const seconds=Math.round(mark.seconds*this.settings.window/147);
      const label=seconds>=60?String(Math.floor(seconds/60))+':'+String(seconds%60).padStart(2,'0'):seconds+'s';
      p.text(label,mark.x+1,1471,2.15,mark.seconds===147?PAL.primary:PAL.medium);
    }
    p.rect(454,1468,119,25,PAL.primary);
    const date=new Date(this.sim.t*1000);
    const clock=[date.getHours(),date.getMinutes(),date.getSeconds()].map(value=>String(value).padStart(2,'0')).join(':');
    p.text(clock,460,1472,2.25,PAL.ink);
    this.hits.push({x:452,y:1460,w:130,h:44,type:'pause',label:this.sim.paused?'Resume':'Pause'});
  }
  drawBottom(){const p=this.p;
    p.line(0,1608,910,1608,PAL.orange,3,.85);
    p.rect(50,1610,128,6,PAL.primary,.9);
    p.rect(512,1610,145,7,PAL.orange,.94);
    p.line(0,1613,909,1613,PAL.subdued,2,.4);
  }
  drawNoise(){const c=this.c;
    c.save();c.globalAlpha=.28;c.drawImage(this.noise,0,0,910,1630);c.restore();
    const step=4;c.save();c.globalAlpha=.07;c.fillStyle='#100601';
    for(let y=0;y<1630;y+=step)c.fillRect(0,y,910,1);
    c.restore();
  }
  drawHotspots(){const p=this.p;for(const h of this.hits){p.line(h.x,h.y,h.x+h.w,h.y,PAL.hot,1,.7);p.line(h.x,h.y,h.x,h.y+h.h,PAL.hot,1,.7)}}
  getHit(x,y){return [...this.hits].reverse().find(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h)}
}

const $ = id => document.getElementById(id);
const settings = {
  channel: "all", window: 147, shyness: 30, labelHold: 2.5, maxCallouts: 2,
  gain: 0, limiter: true, compressor: true, debug: false,
};
const feed = { t: Date.now() / 1000, paused: false, resources: {} };
const canvas = $("scope");
const renderer = new Renderer(canvas, feed, settings);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
let token = new URLSearchParams(location.hash.slice(1)).get("access") || "";
try {
  if (token) sessionStorage.setItem("omni_access", token);
  else token = sessionStorage.getItem("omni_access") || "";
} catch (_) {}
let refreshing = false;
let frame = 0;
let last = 0;
let toastTimer = 0;

function category(event) {
  if (event.category === "memory") return "memory";
  if (event.category === "work") return "tasks";
  if (event.source === "voice-loop status" || event.signal === "presence") return "voice";
  if (event.event === "inference_stage" || ["llm", "perception"].includes(event.signal)) {
    return event.event === "request_received" ? "input" : "inference";
  }
  if (event.category === "tools" || event.signal === "tools") return "tools";
  if (event.signal === "speaking") return "said";
  if (event.signal === "hearing") return "heard";
  if (event.signal === "audio") return event.event === "request_received" ? "input" : "heard";
  if (["vision", "documents"].includes(event.signal)) return "media";
  return "input";
}

function scopeRows(snapshot) {
  const toolStarts = new Map();
  const toolId = event => event.request_id && event.tool_call_id
    ? event.request_id + ":" + event.tool_call_id : null;
  for (const event of snapshot.timeline || []) {
    const id = toolId(event);
    if (id && event.event === "tool_call_started") toolStarts.set(id, Date.parse(event.at));
  }
  const diagnostics = (snapshot.timeline || []).map(event => {
    const at = Date.parse(event.at);
    const invoked = event.event === "tool_call_completed" ? toolStarts.get(toolId(event)) : null;
    const duration = Number.isFinite(event.elapsed_ms) ? event.elapsed_ms
      : Number.isFinite(invoked) && invoked <= at ? at - invoked : null;
    return { ...event, at, channel: category(event), duration };
  });
  const sources = (snapshot.activity_sources?.sources || []).map(event => ({
    id: "source-" + event.id,
    at: Number(event.at) * 1000,
    channel: event.lane === "said" ? "said" : event.lane === "done" ? "tools"
      : event.label === "Typed input" ? "input" : "heard",
    label: event.label, detail: event.text, source: event.authority,
    retention: snapshot.activity_sources.scope, request_id: event.request_id,
    duration: null, audioGenerated: event.audio_generated === true,
  }));
  return [...diagnostics, ...sources].filter(event => Number.isFinite(event.at) && event.at > 0).map(event => {
    const end = event.at / 1000;
    const duration = Number.isFinite(event.duration) ? event.duration : 0;
    const failed = event.status === "error";
    const id = [...String(event.id)].reduce((hash, ch) => Math.imul(hash, 31) + ch.charCodeAt(0) | 0, 0);
    const measure = Number.isFinite(event.duration)
      ? duration >= 1000 ? (duration / 1000).toFixed(1) + "S" : Math.round(duration) + "MS"
      : Number.isInteger(event.tool_round) ? "ROUND " + event.tool_round : event.audioGenerated ? "AUDIO" : "";
    const shorthand = failed ? "ERROR" : event.event === "inference_stage"
      ? event.signal === "perception" ? "PERCEPTION" : event.signal === "speaking" ? "SYNTHESIS" : "LANGUAGE"
      : event.audioGenerated ? "GENERATED" : event.label.split(" ").slice(0, 2).join(" ").slice(0, 10);
    return { ...event, id, key: String(event.id), end, start: end - duration / 1000,
      sig: failed ? 32 : duration ? 26 : 20, failed, dur: duration / 1000, measure, short: shorthand,
      kind: duration ? failed ? "hatched" : "trace" : "pulse" };
  });
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("visible"), 1800);
}

function draw() {
  renderer.render();
}

function tick(timestamp) {
  frame = 0;
  if (document.hidden || reduceMotion.matches || feed.paused) return;
  if (timestamp - last >= 33) {
    feed.t = Date.now() / 1000;
    draw();
    last = timestamp;
  }
  animate();
}

function animate() {
  if (!frame && !document.hidden && !reduceMotion.matches && !feed.paused) {
    frame = requestAnimationFrame(tick);
  }
}

function updateEventOptions() {
  const selected = renderer.selected || "";
  const choices = renderer.allEvents.filter(event => settings.channel === "all" || event.channel === settings.channel);
  const list = $("event-control");
  list.replaceChildren(new Option("Select an event", ""), ...choices.map(event =>
    new Option(CHANNEL_NAMES.get(event.channel) + " · " + event.label, event.key)));
  list.value = selected;
  $("btn-source").disabled = !list.value;
}

function selectChannel(channel) {
  settings.channel = channel;
  renderer.selected = null;
  $("channel-control").value = channel;
  updateEventOptions();
  draw();
}

function selectEvent(id) {
  renderer.selected = id || null;
  $("event-control").value = id || "";
  $("btn-source").disabled = !id;
  draw();
}

function sourceDetails() {
  const event = renderer.allEvents.find(item => item.key === renderer.selected);
  if (!event) return;
  $("event-title").textContent = event.label;
  $("event-source").textContent = event.detail || "";
  const details = [
    ["Channel", CHANNEL_NAMES.get(event.channel)], ["Recorded", new Date(event.at).toLocaleString()],
    ["Source", event.source], ["Retention", event.retention],
    ["Measurement", event.measure], ["Status", event.failed ? "Error" : "Recorded"],
    ["Request", event.request_id], ["Playback", event.audioGenerated ? "Unverified" : ""],
  ];
  $("event-facts").replaceChildren(...details.filter(([,value]) => value).flatMap(([label,value]) => {
    const name = document.createElement("dt");
    const detail = document.createElement("dd");
    name.textContent = label;
    detail.textContent = value;
    return [name, detail];
  }));
  if (!$("event-info").open) $("event-info").showModal();
}

async function refresh(manual = false) {
  if (refreshing || (feed.paused && !manual)) return;
  if (!token) {
    $("sample-status").textContent = "Open the scope with your current access link.";
    if (manual) toast("Access link required");
    $("shell").ariaBusy = "false";
    return;
  }
  refreshing = true;
  $("btn-reset").disabled = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("/api/observatory?activity=1", {
      headers: { Authorization: "Bearer " + token }, cache: "no-store", signal: controller.signal,
    });
    if (!response.ok) throw new Error(response.status === 401 ? "Access link expired" : "Runtime sample unavailable");
    const sample = await response.json();
    if (sample.schema !== "robit.omni.observatory.v1") throw new Error("Unsupported runtime sample");
    renderer.allEvents = scopeRows(sample);
    if (!renderer.allEvents.some(event => event.key === renderer.selected)) renderer.selected = null;
    feed.resources = { ram: sample.live?.environment?.memory?.used_percent, loaded: sample.weights?.loaded };
    feed.t = Date.now() / 1000;
    $("sample-status").textContent = "Sampled " + new Date(sample.captured_at).toLocaleTimeString();
    $("shell").ariaBusy = "false";
    updateEventOptions();
    draw();
    animate();
  } catch (error) {
    $("sample-status").textContent = error.name === "AbortError" ? "Runtime sample timed out" : error.message;
    $("shell").ariaBusy = "false";
    if (manual) toast($("sample-status").textContent);
  } finally {
    clearTimeout(timeout);
    refreshing = false;
    $("btn-reset").disabled = false;
  }
}

function pause() {
  feed.paused = !feed.paused;
  $("paused-badge").hidden = !feed.paused;
  $("btn-pause").setAttribute("aria-label", feed.paused ? "Resume" : "Pause");
  $("btn-pause").textContent = feed.paused ? "▶" : "Ⅱ";
  if (!feed.paused) refresh(true);
  else draw();
}

function settingsPanel(open) {
  $("settings-panel").hidden = !open;
  $("shell").classList.toggle("drawer-open", open);
  (open ? $("btn-close") : $("btn-settings")).focus();
}

function point(event) {
  const bounds = canvas.getBoundingClientRect();
  return { x: (event.clientX - bounds.x) / bounds.width * VIEW.width,
    y: (event.clientY - bounds.y) / bounds.height * VIEW.height };
}

function save(name, href) {
  const link = document.createElement("a");
  link.download = name;
  link.href = href;
  link.click();
}

function screenshot() {
  save("runtime-scope.png", canvas.toDataURL("image/png"));
  toast("Scope image saved");
}

function fullscreen() {
  const action = document.fullscreenElement ? document.exitFullscreen() : $("shell").requestFullscreen();
  action.catch(() => toast("Fullscreen unavailable"));
}

canvas.addEventListener("pointerdown", event => {
  const { x, y } = point(event);
  const hit = renderer.getHit(x, y);
  if (hit?.type === "channel") selectChannel(hit.value === settings.channel ? "all" : hit.value);
  if (hit?.type === "event") selectEvent(hit.value);
  if (hit?.type === "pause") pause();
});
canvas.addEventListener("dblclick", event => {
  const { x, y } = point(event);
  if (renderer.getHit(x, y)?.type === "event") sourceDetails();
});
canvas.addEventListener("pointermove", event => {
  const { x, y } = point(event);
  const hit = renderer.getHit(x, y);
  const hover = $("hover-label");
  hover.hidden = !hit;
  canvas.style.cursor = hit ? "pointer" : "crosshair";
  if (hit) {
    hover.textContent = hit.label;
    hover.style.left = Math.min(window.innerWidth - 150, event.clientX + 12) + "px";
    hover.style.top = Math.min(window.innerHeight - 35, event.clientY + 12) + "px";
  }
});
canvas.addEventListener("pointerleave", () => { $("hover-label").hidden = true; });
for (const [key, label] of CHANNELS) $("channel-control").add(new Option(label, key));
$("channel-control").addEventListener("change", event => selectChannel(event.target.value));
$("window-control").addEventListener("change", event => { settings.window = Number(event.target.value); draw(); });
$("event-control").addEventListener("change", event => selectEvent(event.target.value));
$("btn-source").addEventListener("click", sourceDetails);
$("btn-pause").addEventListener("click", pause);
$("btn-reset").addEventListener("click", () => refresh(true));
$("btn-snapshot").addEventListener("click", screenshot);
$("btn-fullscreen").addEventListener("click", fullscreen);
$("btn-settings").addEventListener("click", () => settingsPanel($("settings-panel").hidden));
$("btn-close").addEventListener("click", () => settingsPanel(false));
$("btn-export").addEventListener("click", () => {
  const rows = renderer.allEvents.map(event => ({
    id: event.key, at: new Date(event.at).toISOString(), channel: event.channel,
    event: event.label, source: event.source, retention: event.retention,
    elapsed_ms: event.duration, failed: event.failed,
  }));
  const url = URL.createObjectURL(new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" }));
  save("runtime-events.json", url);
  toast("Event records saved");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !$("settings-panel").hidden && !$("event-info").open) {
    settingsPanel(false);
    return;
  }
  if ($("event-info").open || ["BUTTON", "SELECT", "INPUT"].includes(event.target.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); pause(); }
  if (event.key.toLowerCase() === "r") refresh(true);
  if (event.key.toLowerCase() === "p") screenshot();
  if (event.key.toLowerCase() === "f") fullscreen();
  if (event.key === "Enter" && event.target === canvas) sourceDetails();
  if (["ArrowUp", "ArrowDown"].includes(event.key) && event.target === canvas) {
    event.preventDefault();
    const index = CHANNELS.findIndex(([key]) => key === settings.channel);
    const step = event.key === "ArrowDown" ? 1 : -1;
    selectChannel(CHANNELS[(index + step + CHANNELS.length) % CHANNELS.length][0]);
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { refresh(); animate(); }
});
reduceMotion.addEventListener("change", () => { draw(); animate(); });
setInterval(() => { if (!document.hidden) refresh(); }, 2000);
draw();
refresh();
animate();
})();
