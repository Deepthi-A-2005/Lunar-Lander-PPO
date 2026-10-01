import math, random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lunar Lander Landing — PPO", page_icon="🚀", layout="wide")

st.markdown("""
<style>
body { background:#05080f; }
.block-container {max-width:1400px;padding-top:2rem}
.hero {padding:2rem;border:1px solid #24313e;border-radius:14px;background:linear-gradient(135deg,#0b121c,#08141a);margin-bottom:1rem}
h1 {font-size:3.2rem!important;line-height:1.05}
.teal {color:#54f0d6}
.card {padding:1rem;border:1px solid #24313e;border-radius:12px;background:#0b111a;margin-bottom:1rem}
.metric {font-family:monospace}
.small {color:#8b9ba6;font-size:.85rem}
.success {color:#54f0d6;font-weight:700}
.fail {color:#ff765c;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="small">🚀 PPO LEARNING DEMO · BROWSER/STREAMLIT SIMULATION</div>
<h1>Teach a spacecraft<br><span class="teal">to land on the Moon.</span></h1>
<p class="small">Train a PPO-style policy, inspect action probabilities, and evaluate safe touchdown between two flags.</p>
</div>
""", unsafe_allow_html=True)

if "lander" not in st.session_state:
    st.session_state.lander={"x":random.uniform(180,780),"y":70,"vx":0.0,"vy":4.0,"fuel":100.0,"angle":0.0}
    st.session_state.episode=1
    st.session_state.history=[]
    st.session_state.running=False

PL, PR, GROUND = 390, 570, 450

def reset_lander():
    st.session_state.lander={"x":random.uniform(180,780),"y":70,"vx":0.0,"vy":4.0,"fuel":100.0,"angle":0.0}

def choose_action(l, ep):
    target=(PL+PR)/2
    dx=target-l["x"]
    desired=max(-24,min(24,dx*.09))
    altitude=GROUND-l["y"]
    safe_vy=42 if altitude>240 else 26 if altitude>110 else 12
    scores={"LEFT":.08,"RIGHT":.08,"UP":.08,"COAST":.20}
    skill=min(.92,.5+ep*.012)
    if l["vx"] > desired+2: scores["LEFT"] += .62*skill
    if l["vx"] < desired-2: scores["RIGHT"] += .62*skill
    if l["vy"] > safe_vy: scores["UP"] += .72*skill
    else: scores["COAST"] += .48
    total=sum(scores.values())
    probs={k:v/total for k,v in scores.items()}
    action=max(probs,key=probs.get)
    if random.random() < max(.02,.22-ep*.004):
        action=random.choice(list(probs))
    return action,probs

def step():
    l=st.session_state.lander
    action,probs=choose_action(l,st.session_state.episode)
    dt=.12
    l["vy"] += 20*dt
    if l["fuel"]>0:
        if action=="UP":
            l["vy"]-=40*dt; l["fuel"]-=7.5*dt
        elif action=="LEFT":
            l["vx"]-=23*dt; l["fuel"]-=3*dt; l["angle"]=max(-.24,l["angle"]-dt)
        elif action=="RIGHT":
            l["vx"]+=23*dt; l["fuel"]-=3*dt; l["angle"]=min(.24,l["angle"]+dt)
        else:
            l["angle"]*=.96
    l["x"]+=l["vx"]*dt
    l["y"]+=l["vy"]*dt
    l["fuel"]=max(0,l["fuel"])
    done=False; success=False
    if l["y"]>=GROUND-24:
        l["y"]=GROUND-24
        success=(PL+15<l["x"]<PR-15 and abs(l["vx"])<11 and l["vy"]<19 and abs(l["angle"])<.22)
        done=True
    if l["x"]<20 or l["x"]>940:
        done=True
    if done:
        velocity=math.hypot(l["vx"],l["vy"])
        reward=round(180+l["fuel"]*.6-velocity if success else -80-velocity*1.4-abs(l["x"]-480)*.08)
        st.session_state.history.append({
            "Episode":st.session_state.episode,
            "Reward":reward,
            "Success":success,
            "Velocity":round(velocity,2),
            "Fuel":round(l["fuel"],1)
        })
        st.session_state.history=st.session_state.history[-30:]
        st.session_state.episode+=1
        reset_lander()
    return action,probs,done,success

def draw_lander(l, action):
    fig,ax=plt.subplots(figsize=(11,5.4))
    fig.patch.set_facecolor("#05080f")
    ax.set_facecolor("#05080f")
    ax.set_xlim(0,960); ax.set_ylim(540,0); ax.axis("off")
    random.seed(10)
    xs=[(i*137.5)%960 for i in range(72)]
    ys=[(i*83.7)%340 for i in range(72)]
    ax.scatter(xs,ys,s=5,c="#d8f1ff",alpha=.45)
    moon=plt.Circle((820,95),38,color="#b6c5cf")
    ax.add_patch(moon)
    ax.fill_between([0,960],[448,448],[540,540],color="#161f2a")
    ground=np.linspace(0,960,200)
    ax.plot(ground,448+np.sin(ground*.037)*14,color="#263a49",lw=1.5)
    ax.fill_between([PL,PR],[GROUND,GROUND],[GROUND-4,GROUND-4],color="#163b39",alpha=.8)
    ax.plot([PL,PR],[GROUND-4,GROUND-4],color="#54f0d6",lw=2)
    for x,d in [(PL,1),(PR,-1)]:
        ax.plot([x,x],[GROUND,GROUND-48],color="white",lw=1.5)
        ax.fill([x,x+d*25,x],[GROUND-47,GROUND-38,GROUND-30],color="#f6bd60")
    # spacecraft
    ax.plot([l["x"]-17,l["x"],l["x"]+17,l["x"]+10,l["x"]-10,l["x"]-17],
            [l["y"]+12,l["y"]-22,l["y"]+12,l["y"]+19,l["y"]+19,l["y"]+12],
            color="#dce9ef",lw=3)
    ax.scatter([l["x"]],[l["y"]-3],s=90,c="#0d6470")
    ax.plot([l["x"]-10,l["x"]-22,l["x"]-27],[l["y"]+15,l["y"]+26,l["y"]+26],color="#b9cad2")
    ax.plot([l["x"]+10,l["x"]+22,l["x"]+27],[l["y"]+15,l["y"]+26,l["y"]+26],color="#b9cad2")
    if action=="UP":
        ax.fill([l["x"]-7,l["x"],l["x"]+7],[l["y"]+20,l["y"]+40,l["y"]+20],color="#f6bd60")
    ax.set_title(f"LUNAR SURFACE / LS-01     Action: {action}",color="#eaf3f7",loc="left",fontsize=11,pad=10)
    return fig

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Training Controls")
    speed=st.select_slider("Training speed",options=[1,2,5,10],value=1)
    manual=st.toggle("Manual control",False)
    if manual:
        st.info("Use the buttons below to control the lander.")
        c1,c2,c3=st.columns(3)
        if c1.button("⬅️"): st.session_state.manual_action="LEFT"
        if c2.button("🔥"): st.session_state.manual_action="UP"
        if c3.button("➡️"): st.session_state.manual_action="RIGHT"
    else:
        st.session_state.manual_action="COAST"
    if st.button("▶️ Start Training",use_container_width=True):
        st.session_state.running=True
    if st.button("⏸️ Pause",use_container_width=True):
        st.session_state.running=False
    if st.button("🔄 Reset",use_container_width=True):
        st.session_state.running=False; st.session_state.episode=1; st.session_state.history=[]; reset_lander()
    st.divider()
    st.caption("This is a PPO-style educational simulation. The displayed loss/entropy metrics are demo telemetry, not a trained neural network.")

# Run simulation steps
action,probs,done,success="COAST",{"LEFT":.1,"RIGHT":.1,"UP":.2,"COAST":.6},False,False
if st.session_state.running:
    for _ in range(speed):
        action,probs,done,success=step()
    st.rerun()
elif manual:
    action=st.session_state.get("manual_action","COAST")
    if action!="COAST":
        # One manual physics step
        l=st.session_state.lander; dt=.12
        l["vy"]+=20*dt
        if l["fuel"]>0:
            if action=="UP": l["vy"]-=40*dt; l["fuel"]-=7.5*dt
            elif action=="LEFT": l["vx"]-=23*dt; l["fuel"]-=3*dt
            elif action=="RIGHT": l["vx"]+=23*dt; l["fuel"]-=3*dt
        l["x"]+=l["vx"]*dt; l["y"]+=l["vy"]*dt
        l["fuel"]=max(0,l["fuel"])
        if l["y"]>=GROUND-24:
            l["y"]=GROUND-24; st.session_state.running=False
            success=PL+15<l["x"]<PR-15 and abs(l["vx"])<11 and l["vy"]<19
            velocity=math.hypot(l["vx"],l["vy"])
            st.session_state.history.append({"Episode":st.session_state.episode,"Reward":round(180+l["fuel"]*.6-velocity if success else -80-velocity*1.4),"Success":success,"Velocity":round(velocity,2),"Fuel":round(l["fuel"],1)})
            st.session_state.episode+=1; reset_lander()

col1,col2=st.columns([2,1])
with col1:
    st.pyplot(draw_lander(st.session_state.lander,action),use_container_width=True)
with col2:
    l=st.session_state.lander
    hist=st.session_state.history
    success_rate=(sum(x["Success"] for x in hist)/len(hist)*100) if hist else 0
    vals=[
        ("EPISODE",st.session_state.episode),
        ("REWARD",hist[-1]["Reward"] if hist else 0),
        ("LANDING VELOCITY",round(math.hypot(l["vx"],l["vy"]),1)),
        ("FUEL",f'{l["fuel"]:.0f}%'),
        ("SUCCESS RATE",f"{success_rate:.0f}%"),
        ("POLICY LOSS",f"{max(.025,.42*math.exp(-st.session_state.episode/25)):.3f}"),
        ("VALUE LOSS",f"{max(.04,.68*math.exp(-st.session_state.episode/30)):.3f}"),
        ("ENTROPY",f"{max(.11,.92*math.exp(-st.session_state.episode/45)):.3f}")
    ]
    for i in range(0,len(vals),2):
        a=col2.columns(2)
        for j in range(2):
            k,v=vals[i+j]
            a[j].metric(k,v)

st.subheader("Policy Output")
pcols=st.columns(4)
for i,k in enumerate(["UP","LEFT","RIGHT","COAST"]):
    pcols[i].metric(k,f"{probs.get(k,0)*100:.0f}%")

st.subheader("Reward Trend")
if st.session_state.history:
    df=pd.DataFrame(st.session_state.history)
    st.line_chart(df.set_index("Episode")["Reward"])
else:
    st.info("Start training to populate the reward history.")

st.subheader("Episode History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history).sort_values("Episode",ascending=False).head(10),use_container_width=True,hide_index=True)
else:
    st.info("No episodes yet.")

st.markdown("---")
st.header("How PPO Learns to Land")
a,b,c,d=st.columns(4)
for box,num,title,text in [
    (a,"01","OBSERVE","Position, velocity, angle and fuel become the state."),
    (b,"02","ACT","The policy assigns probabilities to thrust actions."),
    (c,"03","REWARD","Safe touchdown earns positive reward; crashes are penalized."),
    (d,"04","UPDATE","PPO uses a clipped objective to avoid overly large policy updates.")
]:
    with box:
        st.markdown(f"### {num} · {title}")
        st.write(text)

st.markdown("""
### Mission success criteria
- Touchdown between the two flags
- Low horizontal and vertical velocity
- Stable spacecraft orientation
- Efficient fuel usage

> **Portfolio note:** To make this a *real* PPO implementation, replace the demo policy with a PyTorch/Stable-Baselines3 PPO agent trained on Gymnasium LunarLander and stream its metrics into this Streamlit interface.
""")
