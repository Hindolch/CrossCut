import json,sqlite3,shutil,hashlib,time,sys,zlib,tarfile
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
root=Path('/home/CrossCut')
sys.path.insert(0,str(root/'experiments/LLM4Teach'))
from common import seed_all
from student import Student
from schedules import teacher_weight
from render_training import render_training
out=root/'data/experiments/exp002-astra-seed0'
train=out/'train'
pause=json.loads((out/'pause.json').read_text())
if 'State:\tT' not in Path('/proc/%d/status'%pause['pid']).read_text():
    raise RuntimeError('Trainer must remain stopped throughout snapshot')
config=json.loads((train/'config.json').read_text())
if pause['trainer_status']['updates']!=0:
    raise RuntimeError('Initial-policy reconstruction is valid only before the first optimizer update')
snapshot=out/'pauses'/('step-%07d'%pause['trainer_status']['env_step'])
snapshot.mkdir(parents=True,exist_ok=False)
with sqlite3.connect(str(train/'frames.sqlite')) as source, sqlite3.connect(str(snapshot/'frames.sqlite')) as target:
    source.backup(target)
for name in ('config.json','provenance.json','student.json','teacher.json','teacher_labels.jsonl','episodes.jsonl','metrics.jsonl'):
    if (train/name).exists(): shutil.copy2(train/name,snapshot/name)
shutil.copy2(out/'pause.json',snapshot/'pause.json')
db=sqlite3.connect(str(snapshot/'frames.sqlite'))
rows=db.execute('SELECT step,episode,before_rgb,after_rgb,action,reward,done,info FROM transitions ORDER BY step').fetchall()
labels={r['env_step']:r for r in (json.loads(x) for x in (snapshot/'teacher_labels.jsonl').read_text().splitlines())}
if len(rows)!=pause['trainer_status']['env_step']:
    raise RuntimeError('Pause counter does not match durable transitions')
seed_all(config['seed'],config['device'])
student=Student().to(config['device'])
optimizer=torch.optim.Adam(student.parameters(),lr=config['learning_rate'],eps=1e-5)
hidden=torch.zeros(1,512,device=config['device'])
rollout={k:[] for k in ('image','hidden','action','logprob','value','reward','bootstrap','done','teacher','weight')}
befores=[];afters=[]
for index,row in enumerate(rows):
    step,episode,before,after,action,reward,done,info=row
    if step!=index+1: raise RuntimeError('Transition gap')
    before=np.frombuffer(zlib.decompress(before),dtype=np.uint8).reshape(64,64,3).copy()
    after=np.frombuffer(zlib.decompress(after),dtype=np.uint8).reshape(64,64,3).copy()
    info=json.loads(info)
    with torch.no_grad():
        image=torch.as_tensor(before,device=config['device'])
        logits,value,next_hidden=student(image.unsqueeze(0),hidden)
        distribution=Categorical(logits=logits)
        sampled=distribution.sample()
        if sampled.item()!=action:
            raise RuntimeError('Seeded policy/RNG does not reproduce recorded action at step %d'%step)
        _,next_value,_=student(torch.as_tensor(after,device=config['device']).unsqueeze(0),next_hidden)
    target=torch.tensor(labels[step-1]['probabilities'])
    values={'image':image.cpu(),'hidden':hidden.squeeze(0).cpu(),'action':sampled.squeeze(0).cpu(),
        'logprob':distribution.log_prob(sampled).squeeze(0).cpu(),'value':value.squeeze(0).cpu(),
        'reward':torch.tensor(float(reward)),'bootstrap':(next_value.squeeze(0)*info['discount']).cpu(),
        'done':torch.tensor(float(done)),'teacher':target,'weight':torch.tensor(teacher_weight(config,step-1,True))}
    for key,val in values.items(): rollout[key].append(val)
    hidden=torch.zeros_like(hidden) if done else next_hidden
    befores.append(before);afters.append(after)
torch.save({'student':student.state_dict(),'optimizer':optimizer.state_dict(),'env_step':len(rows),'updates':0,
    'config':config,'hidden_after_executed_actions':hidden.cpu(),'observation_after_executed_actions':afters[-1],
    'partial_rollout':rollout,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
    'numpy_rng':np.random.get_state(),'verification':'Every recorded sampled action matched deterministic reconstruction. No environment actions or optimizer steps executed during reconstruction.',
    'environment_restore':'Live process retains exact environment. For disk recovery replay recorded actions under pinned Crafter and verify all stored RGB frames.'},snapshot/'pause-policy-and-rollout.pt')
np.savez_compressed(snapshot/('trajectory-%07d-%07d.npz'%(1,len(rows))),observations=np.stack(befores),next_observations=np.stack(afters),
    observation_sha256=np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in befores],dtype='S64'),
    next_observation_sha256=np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in afters],dtype='S64'),
    actions=np.array([r[4] for r in rows]),rewards=np.array([r[5] for r in rows]),dones=np.array([r[6] for r in rows]))
video=render_training(snapshot)
metadata={'run_id':'exp002-astra-seed0','phase':'paused','env_step':len(rows),'updates':0,'pid':pause['pid'],
    'snapshot_created_at':time.time(),'source_commit':json.loads((snapshot/'provenance.json').read_text())['git_commit'],
    'policy_reconstruction_verified_actions':len(rows),'new_environment_steps':0,'new_optimizer_steps':0,
    'video':video,'resume_method':'SIGCONT on retained trainer after local teacher/tunnel/monitor restored; do not restart from step zero.'}
(snapshot/'snapshot.json').write_text(json.dumps(metadata,indent=2)+'\n')
files=[{'path':p.name,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(snapshot.iterdir()) if p.is_file()]
(snapshot/'manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
archive=snapshot.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as bundle: bundle.add(snapshot,arcname=snapshot.name)
print(json.dumps({'snapshot':str(snapshot),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'metadata':metadata}))
