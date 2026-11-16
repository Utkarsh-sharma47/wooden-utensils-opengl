"""Real desktop OpenGL wooden utensil study. Python 3.10+.
Run --help for window, headless rendering, camera and shadow comparison modes.
Meshes are generated in code; wood uses CC0 scanned maps. No reference-image texture.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import moderngl
from PIL import Image

ROOT=Path(__file__).resolve().parent

def normalize(v):
    v=np.asarray(v,dtype='f4'); return v / max(float(np.linalg.norm(v)),1e-12)
def look_at(eye,target,up):
    f=normalize(np.array(target)-eye); s=normalize(np.cross(f,up)); u=np.cross(s,f)
    m=np.eye(4,dtype='f4');m[0,:3]=s;m[1,:3]=u;m[2,:3]=-f
    m[:3,3]=-m[:3,:3]@eye;return m

def ortho(l,r,b,t,n,f):
    m=np.eye(4,dtype='f4');m[0,0]=2/(r-l);m[1,1]=2/(t-b);m[2,2]=-2/(f-n)
    m[0,3]=-(r+l)/(r-l);m[1,3]=-(t+b)/(t-b);m[2,3]=-(f+n)/(f-n);return m

def mesh_normals(v,faces):
    v=np.asarray(v,dtype='f4');f=np.asarray(faces,dtype='i4');n=np.zeros_like(v)
    ns=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    for j in range(3):np.add.at(n,f[:,j],ns)
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    return v,n,f

class Mesh:
    def __init__(self,name,v,f,material=0,seed=0,cut=0,model=None):
        self.name=name; self.v,self.n,self.f=mesh_normals(v,f); self.material=material;self.seed=seed;self.cut=cut
        self.model=np.eye(4,dtype='f4') if model is None else model
    def gpu(self,ctx,program,shadow):
        self.buf=ctx.buffer(np.hstack([self.v,self.n]).astype('f4').tobytes());self.idx=ctx.buffer(self.f.astype('i4').tobytes())
        self.vao=ctx.vertex_array(program,[(self.buf,'3f 3f','in_pos','in_normal')],self.idx)
        self.svao=ctx.vertex_array(shadow,[(self.buf,'3f 12x','in_pos')],self.idx)

def transform(x=0,y=0,z=0,angle=0,scale=1):
    a=math.radians(angle);c,s=math.cos(a),math.sin(a)
    m=np.eye(4,dtype='f4');m[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])*scale;m[:3,3]=[x,y,z];return m

def loft(name,sections,mat=0,seed=0,model=None,segments=48):
    # Smooth closed elliptical sections: y, half-width, half-height, centre-z.
    # Catmull-Rom interpolation removes visible kinks along the outline.
    raw=np.asarray(sections,dtype=float); smooth=[]
    for i in range(len(raw)-1):
        p0=raw[max(0,i-1)];p1=raw[i];p2=raw[i+1];p3=raw[min(len(raw)-1,i+2)]
        for t in np.linspace(0,1,8,endpoint=False):
            q=.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)
            q[1:3]=np.maximum(q[1:3],.004);smooth.append(q)
    smooth.append(raw[-1]);sections=smooth
    v=[];f=[]
    for y,w,h,z in sections:
        for j in range(segments):
            a=j/segments*2*np.pi;v.append([w*np.cos(a),y,z+h*np.sin(a)])
    for i in range(len(sections)-1):
        for j in range(segments):
            a=i*segments+j;b=i*segments+(j+1)%segments;c=a+segments;d=b+segments
            f.extend([[a,b,c],[b,d,c]])
    # end caps
    for ring,flip in [(0,True),(len(sections)-1,False)]:
        center=len(v);v.append([0,sections[ring][0],sections[ring][3]])
        for j in range(segments):
            a=ring*segments+j;b=ring*segments+(j+1)%segments
            f.append([center,b,a] if flip else [center,a,b])
    return Mesh(name,v,f,mat,seed,model=model)

def bowl(name,cx,cy,rx,ry,z,depth,thick,mat=0,seed=0,model=None,cut=False):
    # Two continuous dished surfaces joined around a rounded rim.
    seg=96;rings=24;v=[];f=[]
    for side in [0,1]:
        for i in range(rings+1):
            r=max(i/rings,0.0001)
            for j in range(seg):
                a=2*np.pi*j/seg;x=rx*r*np.cos(a);y=ry*r*np.sin(a)
                zz=z+depth*(r**2.2)
                if side:zz-=thick*(0.80+0.20*r)
                v.append([cx+x,cy+y,zz])
        base=side*(rings+1)*seg
        for i in range(rings):
            for j in range(seg):
                a=base+i*seg+j;b=base+i*seg+(j+1)%seg;c=a+seg;d=b+seg
                f.extend([[a,c,b],[b,c,d]] if side else [[a,b,c],[b,d,c]])
    b=(rings+1)*seg
    for j in range(seg):
        a=rings*seg+j;c=rings*seg+(j+1)%seg
        f.extend([[a,a+b,c],[c,a+b,c+b]])
    return Mesh(name,v,f,mat,seed,cut,model)

def sphere(name,center,radii,mat=1,seed=0,seg=24,rings=14):
    v=[];f=[]
    for i in range(rings+1):
        t=np.pi*(i+.001)/(rings+.002)
        for j in range(seg):
            p=j*2*np.pi/seg
            v.append(np.array(center)+np.array(radii)*[np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    for i in range(rings):
        for j in range(seg):
            a=i*seg+j;b=i*seg+(j+1)%seg;c=a+seg;d=b+seg;f.extend([[a,b,c],[b,d,c]])
    return Mesh(name,v,f,mat,seed)

def plane():return Mesh('beige tabletop',[[-12,-12,0],[12,-12,0],[12,12,0],[-12,12,0]],[[0,1,2],[0,2,3]],1)

def make_scene(scene_kind="full"):
    out=[plane()]
    # Positions are editable. Alternating directions follow the supplied image.
    kinds=['fork','spoon','spatula','ladle','turner','perforated']
    positions=[(-2.75,-.02,180),(-1.65,.12,0),(-.55,-.01,180),(.55,.12,0),(1.65,-.04,180),(2.75,.09,0)]
    for k,(kind,(x,y,ang)) in enumerate(zip(kinds,positions)):
        m=transform(x,y,0,ang);m[:3,0]*=.80;m[:3,1]*=1.18
        seed=k*3.71+.4
        sec=[(-2.13,.025,.032,.12),(-2.09,.085,.064,.12),(-1.95,.115,.068,.12),(-1.55,.105,.065,.13),(-1.,.088,.064,.145),(-.4,.071,.063,.17),(.15,.082,.064,.205),(.56,.11,.048,.198),(.72,.12,.018,.12)]
        out.append(loft(kind+' handle',sec,seed=seed,model=m))
        if kind in ['spoon','ladle','perforated']:
            rx=.40 if kind!='ladle' else .46;ry=.64 if kind!='ladle' else .56
            out.append(bowl(kind+' head',0,1.18,rx,ry,.135,.17 if kind!='ladle' else .24,.060,seed=seed,model=m,cut=1 if kind=='perforated' else 0))
            if kind=='perforated':
                # Inner hole walls connect the fragment-cut top and bottom surfaces.
                v=[];f=[];ns=32
                for row in range(4):
                    for col in range(3):
                        cx=(col-1)*.205;cy=.87+row*.205;base=len(v)
                        for side in [0,1]:
                            for j in range(ns):
                                a=j*2*np.pi/ns;xx=cx+.047*np.cos(a);yy=cy+.066*np.sin(a)
                                rr=np.sqrt((xx/rx)**2+((yy-1.18)/ry)**2)
                                zz=.135+.17*rr**2.2-side*.06*(.8+.2*rr)
                                v.append([xx,yy,zz])
                        for j in range(ns):
                            q=base+j;b=base+(j+1)%ns;f.extend([[q,b,q+ns],[b,b+ns,q+ns]])
                out.append(Mesh('perforated spoon hole walls',v,f,seed=seed,model=m))

        elif kind in ['spatula','turner']:
            w=.38 if kind=='spatula' else .33
            sec=[(.48,.09,.055,.21),(.72,.18,.057,.18),(1.0,w*.87,.045,.14),(1.35,w,.040,.105),(1.68,w*.98,.028,.08),(1.79,w*.90,.018,.07),(1.815,w*.72,.007,.068)]
            out.append(loft(kind+' blade',sec,seed=seed,model=m))
        else:
            head=loft('fork carved head',[(.53,.11,.05,.20),(.85,.23,.05,.16),(1.1,.36,.045,.12),(1.45,.36,.038,.105),(1.73,.33,.026,.12),(1.86,.29,.010,.14)],seed=seed,model=m)
            head.cut=2;out.append(head)
    # Pepper bowl clipped by the lower-left crop. Seeded placement is repeatable.
    out.append(bowl('earthenware pepper bowl',-3.66,-2.72,.66,.66,.07,.30,.09,mat=2))
    rng=np.random.default_rng(26)
    for i in range(110):
        r=.52*np.sqrt(rng.random());a=rng.uniform(0,2*np.pi)
        xx=-3.66+r*np.cos(a);yy=-2.72+r*np.sin(a)
        out.append(sphere('peppercorn',(xx,yy,.19+rng.random()*.06),(.045,.045,.041),3,float(i),seg=12,rings=8))
    for i in range(18):
        xx=-3.10+rng.uniform(-.2,.8); yy=-3.05+rng.uniform(-.4,.35)
        out.append(sphere('loose peppercorn',(xx,yy,.04),(.038,.038,.038),3,float(i),seg=12,rings=8))
    # Cloth corner, generated woven surface with subtle folds.
    v=[];f=[];N=45
    for i in range(N):
        for j in range(N):
            x=-4.7+i/(N-1)*2.1;y=2.05+j/(N-1)*2.7
            zz=.035+.065*np.sin((x+y)*7)**2+.022*np.sin(y*18+x*4)
            v.append([x+.23*(y-2.0),y,zz])
    for i in range(N-1):
        for j in range(N-1):
            a=i*N+j;f.extend([[a,a+N,a+1],[a+1,a+N,a+N+1]])
    out.append(Mesh('folded linen',v,f,4))
    # Rolled bark strips: open spiral cross sections, not solid cylinders.
    for i,(x,y,a) in enumerate([(2.40,-3.18,62),(2.74,-2.85,58),(3.24,-3.20,58)]):
        v=[];f=[];ns=130;ny=18
        for side in range(2):
            for yi in range(ny+1):
                yy=-.65+1.30*yi/ny
                for j in range(ns):
                    t=j/(ns-1);ang=t*4.8*np.pi
                    r=.035+.11*t+side*.010+.004*np.sin(yy*24+ang*2)
                    v.append([r*np.cos(ang),yy,.17+r*np.sin(ang)])
        stride=(ny+1)*ns
        for side in range(2):
            for yi in range(ny):
                for j in range(ns-1):
                    q=side*stride+yi*ns+j
                    f.extend([[q,q+1,q+ns],[q+1,q+ns+1,q+ns]])
        for end in [0,ny]:
            for j in range(ns-1):
                q=end*ns+j;f.extend([[q,q+stride,q+1],[q+1,q+stride,q+stride+1]])
        out.append(Mesh('rolled cinnamon bark',v,f,5,i,model=transform(x,y,0,a)))
    # Small compound herb sprigs: lobed leaf meshes, curved surfaces and thin stems.
    def leaf(cx,cy,angle,size):
        v=[];f=[];segs=48;nr=8
        for i in range(nr+1):
            r=max(i/nr,.0001)
            for j in range(segs):
                a=2*np.pi*j/segs
                lobe=.80+.16*np.cos(5*a)+.04*np.cos(15*a)
                x=size*r*lobe*np.cos(a);y=size*r*lobe*np.sin(a)*1.25
                z=.018+.045*(1-r*r)+.022*np.sin(a*2)*r*r
                v.append([x,y,z])
        for i in range(nr):
            for j in range(segs):
                q=i*segs+j;n=i*segs+(j+1)%segs
                f.extend([[q,n,q+segs],[n,n+segs,q+segs]])
        return Mesh('coriander leaf',v,f,6,model=transform(cx,cy,.025,angle))
    for cx,cy in [(2.45,3.42),(3.18,3.12),(3.65,-3.25)]:
        stem=loft('herb stem',[(-.50,.012,.012,.06),(.0,.012,.012,.065),(.50,.008,.008,.08)],mat=6,model=transform(cx,cy,0,-20),segments=12)
        out.append(stem)
        for t in range(3):
            yy=cy+(t-1)*.31;xx=cx+.08*(t-1)
            for side in [-1,1]:out.append(leaf(xx+side*.20,yy,(t*31+side*27),.19+.035*t))
    out.append(bowl('partial clay bowl',3.90,3.15,.58,.58,.05,.32,.07,2))
    if scene_kind == 'single':
        out=[o for o in out if o.name in ['beige tabletop','spoon handle','spoon head']]
        for o in out:
            if o.material==0:
                o.model=transform(0,.15,0,-18)
                o.model[:3,0]*=.80;o.model[:3,1]*=1.18
    return out

VERT='''#version 330
in vec3 in_pos; in vec3 in_normal;
uniform mat4 model,vp,light_vp;
out vec3 world,normal,local;out vec4 lightpos;
void main(){vec4 p=model*vec4(in_pos,1);world=p.xyz;local=in_pos;normal=mat3(transpose(inverse(model)))*in_normal;lightpos=light_vp*p;gl_Position=vp*p;}
'''
SVERT='''#version 330
in vec3 in_pos;uniform mat4 model,light_vp;out vec3 local;
void main(){local=in_pos;gl_Position=light_vp*model*vec4(in_pos,1);}
'''
CUT='''
bool hole(vec3 p,int kind){
 if(kind==2){for(int i=0;i<3;i++){float cx=(float(i)-1.)*.19; if(abs(p.x-cx)<.040 && p.y>1.20)return true;}return false;}
 for(int row=0;row<4;row++) for(int col=0;col<3;col++){
 vec2 c=vec2((float(col)-1.)*.205, .87+float(row)*.205);
 if(length((p.xy-c)/vec2(.047,.066))<1.) return true;
 }return false;}
'''
SFRAG='#version 330\nin vec3 local;uniform int cut;\n'+CUT+'\nvoid main(){if(cut>0 && hole(local,cut))discard;}\n'
FRAG='''#version 330
in vec3 world,normal,local;in vec4 lightpos;out vec4 color;
uniform vec3 eye,light;uniform sampler2D shadowmap,wood_color,wood_rough,wood_normal;uniform int material,shadow_mode;
uniform float seed,exposure;uniform int cut;
const float PI=3.14159265359;
'''+CUT+'''
float hash(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float noise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);}
float fbm(vec3 p){return .55*noise(p)+.28*noise(p*2.03)+.13*noise(p*4.07)+.04*noise(p*8.1);}
float grain(vec3 p){
 p.y+=seed;float warp=fbm(vec3(p.x*5.,p.y*.45,p.z*6.));
 float ring=sin(p.x*130.+warp*17.+sin(p.y*1.6)*2.5);
 float fine=sin(p.x*520.+warp*24.+noise(p*vec3(20.,1.,20.))*8.);
 return .55+.19*ring+.09*fine+.15*fbm(p*vec3(65.,3.,20.));
}
float visibility(vec3 N,vec3 L){
 vec3 q=lightpos.xyz/lightpos.w*.5+.5;if(q.z>1.||q.x<0.||q.x>1.||q.y<0.||q.y>1.)return 1.;
 float bias=max(.00027*(1.-dot(N,L)),.000065);vec2 tex=1./vec2(textureSize(shadowmap,0));float sum=0.;
 if(shadow_mode==0)return q.z-bias>texture(shadowmap,q.xy).r?0.:1.;
 // 49-tap PCF. Stable edge smoothing, not a physically exact area-light shadow.
 for(int x=-3;x<=3;x++)for(int y=-3;y<=3;y++)sum+=q.z-bias>texture(shadowmap,q.xy+vec2(x,y)*tex*2.1).r?0.:1.;
 return sum/49.;
}
vec3 fresnel(float x,vec3 f){return f+(1.-f)*pow(clamp(1.-x,0.,1.),5.);}
vec3 filmic(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
void main(){
 if(cut>0&&hole(local,cut))discard;
 vec3 N=normalize(normal);if(!gl_FrontFacing)N=-N;
 vec3 base;float rough=.56;float detail=0.;
 if(material==0){
  vec2 uv=vec2(local.x*.36+seed*.137,local.y*.19+seed*.063);
  vec3 tex=pow(texture(wood_color,uv).rgb,vec3(2.2));
  // Warm the scanned walnut-like wood toward a lighter oiled-wood finish.
  base=pow(tex,vec3(.58))*vec3(1.60,1.13,.64);
  float micro=texture(wood_rough,uv).r;
  rough=clamp(.26+micro*.24,.30,.57);
  detail=texture(wood_normal,uv).r*.10;

 }else if(material==1){float n=fbm(world*32.);base=vec3(.69,.60,.46)*(.92+.12*n);rough=.91;detail=n*.07;}
 else if(material==2){float n=noise(world*80.);base=vec3(.39,.135,.055)*(.80+.3*n);rough=.76;detail=n*.12;}
 else if(material==3){float n=noise(local*100.);base=mix(vec3(.025,.016,.009),vec3(.14,.085,.031),n);rough=.94;detail=n*.2;}
 else if(material==4){float weave=sin(world.x*650.)*sin(world.y*650.);base=vec3(.65,.59,.46)*(1.+.04*weave);rough=.96;detail=weave*.045;}
 else if(material==5){float n=sin(local.y*8.+noise(local*vec3(70.,1.,70.))*3.);base=mix(vec3(.18,.045,.009),vec3(.42,.17,.045),n*.5+.5);rough=.86;detail=n*.15;}
 else{float n=noise(local*80.);float vein=pow(abs(sin(local.x*130.+local.y*75.)),18.);base=vec3(.040,.12,.008)*(.8+.3*n)+vec3(.025,.035,.004)*vein;rough=.83;detail=n*.06;}
 vec3 dpdx=dFdx(world),dpdy=dFdy(world);vec3 grad=cross(dpdy,N)*dFdx(detail)+cross(N,dpdx)*dFdy(detail);
 N=normalize(N-grad*.07);
 if(material==0){
  vec2 uv=vec2(local.x*.36+seed*.137,local.y*.19+seed*.063);
  vec2 dx=dFdx(uv),dy=dFdy(uv);
  vec3 T=normalize(dpdx*dy.y-dpdy*dx.y),B=normalize(-dpdx*dy.x+dpdy*dx.x);
  vec3 mn=texture(wood_normal,uv).xyz*2.-1.;
  N=normalize(N+(T*mn.x+B*mn.y)*.16);
 }

 vec3 V=normalize(eye-world),L=normalize(light-world),H=normalize(V+L);float nl=max(dot(N,L),0.),nv=max(dot(N,V),.0001),nh=max(dot(N,H),0.);
 float a=rough*rough,a2=a*a,den=nh*nh*(a2-1.)+1.;float D=a2/(PI*den*den+.0001);
 float k=(rough+1.)*(rough+1.)/8.;float G=(nv/(nv*(1.-k)+k))*(nl/(nl*(1.-k)+k));vec3 F=fresnel(max(dot(H,V),0.),vec3(.04));
 vec3 spec=D*G*F/(4.*nv*max(nl,.001)+.0001);
 float sh=visibility(N,L);vec3 direct=((1.-F)*base/PI+spec)*nl*vec3(3.7,3.35,2.85)*sh;
 // Analytic hemispheric environment fill. This is not image-based HDR lighting.
 float hemi=.5+.5*N.z;vec3 env=mix(vec3(.22,.18,.13),vec3(.50,.55,.58),hemi);
 vec3 ambient=base*env*.62;
 vec3 c=filmic((direct+ambient)*exposure);c=pow(c,vec3(1./2.2));color=vec4(c,1);
}
'''

class Renderer:
    def __init__(self,ctx,width,height,shadow_mode=1,scene_kind="full"):
        self.ctx=ctx;self.w=width;self.h=height;self.shadow_mode=shadow_mode
        self.prog=ctx.program(vertex_shader=VERT,fragment_shader=FRAG);self.sp=ctx.program(vertex_shader=SVERT,fragment_shader=SFRAG)
        self.wood=[]
        for i,name in enumerate(['wood-color.jpg','wood-rough.jpg','wood-normal.jpg']):
            im=Image.open(ROOT/name).convert('RGB').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            tex=ctx.texture(im.size,3,im.tobytes());tex.build_mipmaps();tex.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR);tex.anisotropy=min(8.,ctx.max_anisotropy);self.wood.append(tex)
        self.scene_kind=scene_kind;self.objects=make_scene(scene_kind)
        for m in self.objects:m.gpu(ctx,self.prog,self.sp)
        self.depth=ctx.depth_texture((2048,2048));self.depth.compare_func='';self.depth.repeat_x=False;self.depth.repeat_y=False
        self.sfb=ctx.framebuffer(depth_attachment=self.depth)
        self.tex=ctx.texture((width,height),4);self.z=ctx.depth_renderbuffer((width,height));self.fb=ctx.framebuffer([self.tex],self.z)
        self.samples=min(4,ctx.max_samples)
        self.ms_color=ctx.renderbuffer((width,height),4,samples=self.samples)
        self.ms_depth=ctx.depth_renderbuffer((width,height),samples=self.samples)
        self.msfb=ctx.framebuffer([self.ms_color],self.ms_depth)
        self.light=np.array([-3.8,4.6,8.5],dtype='f4')
        self.lv=ortho(-6.5,6.5,-6.5,6.5,.1,25.)@look_at(self.light,np.array([0.,0.,0.]),[0,1,0])
        ctx.enable(moderngl.DEPTH_TEST)
    def render(self,tilt=0,exposure=1.0):
        eye=np.array([0.,-tilt,10.],dtype='f4');self.eye=eye
        # Square target with additional horizontal view on wide windows.
        halfy=2.8 if self.scene_kind=='single' else 3.85;halfx=halfy*self.w/self.h
        vp=ortho(-halfx,halfx,-halfy,halfy,.1,30.)@look_at(eye,np.array([0.,0.,.1]),[0,1,0])
        self.sfb.use();self.ctx.viewport=(0,0,2048,2048);self.sfb.clear(depth=1.)
        self.sp['light_vp'].write(self.lv.T.astype('f4').tobytes())
        for o in self.objects:
            self.sp['model'].write(o.model.T.astype('f4').tobytes());self.sp['cut'].value=o.cut;o.svao.render()
        self.msfb.use();self.ctx.viewport=(0,0,self.w,self.h);self.msfb.clear(.7,.62,.5,1,depth=1.)
        p=self.prog;p['vp'].write(vp.T.astype('f4').tobytes());p['light_vp'].write(self.lv.T.astype('f4').tobytes());p['eye'].value=tuple(eye);p['light'].value=tuple(self.light)
        p['shadowmap'].value=0;self.depth.use(0);
        for i,name in enumerate(['wood_color','wood_rough','wood_normal']):p[name].value=i+1;self.wood[i].use(i+1)
        p['shadow_mode'].value=self.shadow_mode;p['exposure'].value=exposure
        for o in self.objects:
            p['model'].write(o.model.T.astype('f4').tobytes());p['material'].value=o.material;p['seed'].value=o.seed;p['cut'].value=o.cut;o.vao.render()
        self.ctx.copy_framebuffer(self.fb,self.msfb)
    def save(self,path):
        Image.frombytes('RGB',(self.w,self.h),self.fb.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path)
    def info(self):
        return {'scene':self.scene_kind,'opengl':self.ctx.info['GL_VERSION'],'renderer':self.ctx.info['GL_RENDERER'],'mesh_count':len(self.objects),'triangles':sum(len(x.f) for x in self.objects),'resolution':[self.w,self.h],'shadows':'PCF' if self.shadow_mode else 'hard','lighting':'GGX direct light + analytic hemisphere fill, no HDRI','generated_meshes':True,'wood_texture':'Poly Haven fine_grained_wood, CC0','tested_on_user_laptop':False}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--render',metavar='PNG',help='Render one actual OpenGL frame and exit (EGL by default).');ap.add_argument('--backend',default='egl',choices=['egl','default']);ap.add_argument('--width',type=int,default=1200);ap.add_argument('--height',type=int,default=1200);ap.add_argument('--tilt',type=float,default=0,help='Oblique camera offset; try 4.');ap.add_argument('--shadow',choices=['hard','pcf'],default='pcf');ap.add_argument('--exposure',type=float,default=1.0);ap.add_argument('--stats',help='Write renderer details as JSON');ap.add_argument('--frames',type=int,default=0,help='Exit window after N frames (smoke testing).');ap.add_argument('--scene',choices=['full','single'],default='full',help='Full tabletop or one isolated wooden spoon.');args=ap.parse_args()
    if args.width<64 or args.height<64 or max(args.width,args.height)>4096:ap.error('width/height must be between 64 and 4096')
    if args.render:
        try:ctx=moderngl.create_standalone_context(require=330,**({'backend':'egl'} if args.backend=='egl' else {}))
        except Exception as e:raise SystemExit('Cannot create OpenGL 3.3 context. Try interactive mode with a working graphics driver or --backend default. Details: '+str(e))
        r=Renderer(ctx,args.width,args.height,args.shadow=='pcf',args.scene);start=time.perf_counter();r.render(args.tilt,args.exposure);ctx.finish();elapsed=time.perf_counter()-start;r.save(args.render);info=r.info();info['first_frame_seconds']=round(elapsed,3)
        print(json.dumps(info,indent=2));print('Saved:',args.render)
        if args.stats:Path(args.stats).write_text(json.dumps(info,indent=2)+'\n')
    else:
        import glfw
        if not glfw.init():raise SystemExit('GLFW could not open a display. On headless Linux use --render preview.png.')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3);glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3);glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE);glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True);glfw.window_hint(glfw.RESIZABLE,False)
        win=glfw.create_window(min(args.width,1000),min(args.height,1000),'Wooden Utensils | OpenGL | S shadows, T camera, P screenshot, Esc quit',None,None)
        if not win:glfw.terminate();raise SystemExit('OpenGL 3.3 window creation failed. Check the graphics driver.')
        glfw.make_context_current(win);glfw.swap_interval(1);ctx=moderngl.create_context(require=330);w,h=glfw.get_framebuffer_size(win);r=Renderer(ctx,w,h,args.shadow=='pcf',args.scene);state={'tilt':args.tilt}
        def key(win,key,scan,action,mods):
            if action!=glfw.PRESS:return
            if key==glfw.KEY_ESCAPE:glfw.set_window_should_close(win,True)
            if key==glfw.KEY_S:r.shadow_mode=1-r.shadow_mode
            if key==glfw.KEY_T:state['tilt']=4 if state['tilt']==0 else 0
            if key==glfw.KEY_P:r.save('screenshot.png');print('Saved screenshot.png')
        glfw.set_key_callback(win,key);print(json.dumps(r.info(),indent=2))
        frame_count=0
        while not glfw.window_should_close(win):
            r.render(state['tilt'],args.exposure);ctx.copy_framebuffer(ctx.screen,r.fb);glfw.swap_buffers(win);glfw.poll_events();frame_count+=1
            if args.frames and frame_count>=args.frames:glfw.set_window_should_close(win,True)
        glfw.terminate()
if __name__=='__main__':main()
