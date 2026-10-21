// The Kalki RP1 robot, built from the converter's own model data: a body tree for forward
// kinematics and one convex support hull per body for shape. Those hulls are approximate volumes,
// not the production shells -- they are what g1_to_kalki.py ships and what it uses for framing.
export async function loadKalki(THREE, base) {
  const get = (name) => fetch(new URL(name, base)).then((r) => {
    if (!r.ok) throw new Error(`${name}: ${r.statusText}`);
    return r.json();
  });
  const [model, hulls] = await Promise.all([get('kalki/model.json'), get('kalki/hulls.json')]);

  const material = new THREE.MeshStandardMaterial({
    color: 0xe3e5e8, roughness: 0.5, metalness: 0.05, flatShading: true, side: THREE.DoubleSide,
  });
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
    for (const part of hulls[body.name] || []) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(part.v.flat(), 3));
      geometry.setIndex(part.f.flat());
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      node.add(mesh);
    }
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
