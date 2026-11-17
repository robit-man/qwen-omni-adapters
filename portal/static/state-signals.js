/* Source-time rails for authorized speech, typed input, actions and replies. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id), NS = "http://www.w3.org/2000/svg";
  const make = (tag, cls, text) => { const node = document.createElement(tag); node.className = cls; if (text !== undefined) node.textContent = text; return node; };
  let windowSeconds = 300, sample = null, fingerprint = "";
  const observer = new ResizeObserver(() => requestAnimationFrame(draw));
  function svg(tag, attrs) { const node = document.createElementNS(NS,tag); Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value))); return node; }
  function draw() {
    const now = Date.now()/1000, instrument = $("state-signal-lanes"), wires = instrument?.querySelector("svg");
    if (!instrument || !wires) return;
    const root = instrument.getBoundingClientRect(), width = instrument.clientWidth, height = instrument.clientHeight, rail = height-28;
      wires.setAttribute("viewBox",`0 0 ${width} ${height}`);
      const paths = [svg("path",{d:`M0,${rail}H${width}`})];
      for (let index=0;index<=10;index++) paths.push(svg("path",{d:`M${width*index/10},${rail-4}v8`}));
      instrument.querySelectorAll(".signal-source-card").forEach(card => {
        const bounds = card.getBoundingClientRect(), x = bounds.left-root.left+bounds.width/2, y = bounds.bottom-root.top;
        const viewport = card.closest(".signal-lane-scroll").getBoundingClientRect();
        if (bounds.right <= viewport.left || bounds.left >= viewport.right) return;
        const sourceX = Math.max(3, Math.min(width-3, (Number(card.dataset.at)-(now-windowSeconds))/windowSeconds*width));
        paths.push(svg("path",{d:`M${sourceX},${rail}V${y+14}H${x}V${y}`}));
        paths.push(svg("rect",{x:sourceX-2,y:rail-2,width:4,height:4,class:"source-junction"}));
      });
      wires.replaceChildren(...paths);
  }
  function render(value) {
    if (!$("state-signal-lanes") || !value) return;
    sample = value;
    const now = Date.now()/1000, sources = (value.sources || []).filter(item=>item.at>=now-windowSeconds&&item.at<=now+1);
    const key = JSON.stringify([windowSeconds,value.scope,sources.map(item=>[item.id,item.text,item.audio_generated,item.addressee,item.authority])]);
    $("state-signal-scope").textContent = `${value.scope || "This browser"} · ${sources.length} matching source records · bounded source-time view`;
    if (key === fingerprint) { requestAnimationFrame(draw); return; }
    fingerprint = key;
    const scroll = new Map([...document.querySelectorAll(".signal-lane-scroll")].map(node=>[node.dataset.lane,{left:node.scrollLeft,end:node.scrollWidth-node.clientWidth-node.scrollLeft<20}]));
    observer.disconnect();
    const lanes = [["heard","HEARD","No speech transcript or typed input in this window."],["done","DONE","No recorded tool returns in this window."],["said","SAID","No completed reply text in this window."]].map(([lane,label,empty])=>{
      const section = make("section","state-signal-lane",""), heading = make("h4","signal-lane-name",label);
      const viewport = make("div","signal-lane-scroll",""), runner = make("div","signal-runner",""), cards = make("ol","signal-source-cards","");
      viewport.dataset.lane=lane;viewport.tabIndex=0;viewport.setAttribute("role","region");viewport.setAttribute("aria-label",`${label.toLowerCase()} source timeline`);
      const rows = sources.filter(item=>item.lane===lane).slice(0,3).reverse();
      runner.style.setProperty("--source-count",String(Math.max(1,rows.length)));
      for (const item of rows) {
        const card = make("li","signal-source-card frame","");card.dataset.at=String(item.at);
        const header = make("div","signal-source-heading","");header.append(make("span","",item.label),make("time","",new Date(item.at*1000).toLocaleTimeString()));
        const text=make("p","signal-source-text",item.text);text.tabIndex=0;
        const attribution = lane==="heard" ? `${item.authority} · addressee ${item.addressee || "UNKNOWN"}` : lane==="said" ? `${item.audio_generated?"Audio generated":"Text response"} · playback unverified` : "Tool return · source receipt";
        const detail = make("details","signal-source-detail","");detail.append(make("summary","","Source"),make("p","",`Source ${item.id} · ${item.actor}\nRequest ${item.request_id}\nSHA-256 ${item.source_sha256}${item.text_truncated?"\nLong source excerpt is truncated.":""}`));
        card.append(header,text,make("p","signal-source-authority",attribution),detail);cards.append(card);
      }
      if (!rows.length) cards.append(make("li","signal-source-empty",empty));
      runner.append(cards);viewport.append(runner);section.append(heading,viewport);
      viewport.addEventListener("scroll",()=>requestAnimationFrame(draw),{passive:true});
      return section;
    });
    const axis = make("div","signal-axis-labels","");axis.append(make("span","",`−${Math.round(windowSeconds/60)} min`),make("span","","NOW"));
    $("state-signal-lanes").replaceChildren(svg("svg",{class:"signal-source-wires","aria-hidden":"true"}),...lanes,axis);
    observer.observe($("state-signal-lanes"));
    document.querySelectorAll(".signal-lane-scroll").forEach(node=>{const old=scroll.get(node.dataset.lane);node.scrollLeft=!old||old.end?node.scrollWidth:old.left;});
    requestAnimationFrame(draw);
  }
  document.querySelectorAll("[data-signal-window]").forEach(button=>button.addEventListener("click",()=>{
    windowSeconds=Number(button.dataset.signalWindow);fingerprint="";
    document.querySelectorAll("[data-signal-window]").forEach(item=>item.setAttribute("aria-pressed",String(item===button)));
    if(sample)render(sample);
  }));
  document.fonts?.ready.then(draw);
  window.OmniStateSignals={render,clear(){sample=null;fingerprint="";$("state-signal-lanes")?.replaceChildren();}};
})();
