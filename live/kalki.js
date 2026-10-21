// The Kalki RP1: a body tree for forward kinematics (live/kalki/model.json, from the converter's own model data)
// with the production meshes hung on it. The meshes are live/kalki/kalki.glb: the MuJoCo model's ~1.6M triangles
// decimated to ~270k, with their textures baked into vertex colours (tools/build_kalki_glb.py). If that file cannot be
// loaded the figure falls back to one convex support hull per body (live/kalki/hulls.json), the coarse volumes
// g1_to_kalki.py ships for framing.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export async function loadKalki(THREE, base) {
  const url = (name) => new URL(name, base).href;
  const getJson = (name) => fetch(url(name)).then((r) => {
    if (!r.ok) throw new Error(`${name}: ${r.statusText}`);
    return r.json();
  });
  const model = await getJson('kalki/model.json');

  const nodes = new Map();
  const root = new THREE.Group();
  const hinges = new Map();

  for (const body of model.bodies) {
    if (!body.parent) { nodes.set(body.name, root); continue; }   // the world body
    const node = new THREE.Group();
    node.position.set(body.pos[0], body.pos[1], body.pos[2]);     // every body quat is identity
    (nodes.get(body.parent) || root).add(node);
    nodes.set(body.name, node);
    if (body.joint) {
      hinges.set(body.joint, { node, axis: new THREE.Vector3(...body.axis).normalize() });
    }
  }

  // Each glTF node is one visual mesh with its vertices already in its body's frame (userData.body).
  async function addMeshes() {
    const gltf = await new GLTFLoader().loadAsync(url('kalki/kalki.glb'));
    let attached = 0;
    for (const object of [...gltf.scene.children]) {
      const parent = nodes.get(object.userData.body);
      if (!parent) continue;
      object.traverse((o) => { if (o.isMesh) o.castShadow = true; });
      parent.add(object);
      attached++;
    }
    if (!attached) throw new Error('no mesh matched a body in the tree');
  }

  async function addHulls() {
    const hulls = await getJson('kalki/hulls.json');
    const material = new THREE.MeshStandardMaterial({
      color: 0xe3e5e8, roughness: 0.5, metalness: 0.05, flatShading: true, side: THREE.DoubleSide,
    });
    for (const [name, parts] of Object.entries(hulls)) {
      const node = nodes.get(name);
      if (!node) continue;
      for (const part of parts) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(part.v.flat(), 3));
        geometry.setIndex(part.f.flat());
        geometry.computeVertexNormals();
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        node.add(mesh);
      }
    }
  }

  try {
    await addMeshes();
  } catch (e) {
    console.warn('Kalki meshes did not load; drawing the support hulls instead.', e);
    await addHulls();
  }

  const order = model.joint_order.map((name) => hinges.get(name));
  if (order.some((h) => !h)) throw new Error('Kalki model is missing a joint named in joint_order');
  const pelvis = nodes.get('base_link');

  return {
    object: root,
    follow: pelvis,                       // what the camera tracks
    jointCount: order.length,
    // qpos: 3 root position, 4 root quaternion (wxyz), then one angle per joint in joint_order
    setPose(q) {
      if (!q) return;
      pelvis.position.set(q[0], q[1], q[2]);
      pelvis.quaternion.set(q[4], q[5], q[6], q[3]);
      for (let i = 0; i < order.length; i++) {
        order[i].node.setRotationFromAxisAngle(order[i].axis, q[7 + i] || 0);
      }
    },
    rest() { this.setPose([0, 0, 0.898, 1, 0, 0, 0, ...new Array(order.length).fill(0)]); },
  };
}
