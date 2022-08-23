import './style.css'

import * as THREE from './node_modules/three/src/Three.js';
import { OrbitControls } from './node_modules/three/examples/jsm/controls/OrbitControls';

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer({
    canvas: document.querySelector('#bg'),
});

renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
camera.position.setZ(30);

renderer.render(scene, camera);

const geometry = new THREE.TorusGeometry(10, 3, 15, 100);
const material = new THREE.MeshStandardMaterial({ color: 0xcc0000, wireframe: true });
const torus = new THREE.Mesh(geometry, material);

scene.add(torus);

const pointLight = new THREE.PointLight(0xffffff);
pointLight.position.set(5, 5, 5);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(pointLight, ambientLight);

const gridHelper = new THREE.GridHelper(200, 50);
scene.add(gridHelper);

const controls = new OrbitControls(camera, renderer.domElement);

const starGeometry = new THREE.SphereGeometry(0.25, 24, 24);
const starMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
function addStart() {
  const star = new THREE.Mesh(starGeometry, starMaterial);
  const [x, y, z] = Array(3).fill().map(() => THREE.MathUtils.randFloatSpread(100));
  star.position.set(x, y, z);
  scene.add(star);
}

Array(200).fill().forEach(addStart);

const spaceTexture = new THREE.TextureLoader().load('./assets/space.jpg');
scene.background = spaceTexture;

const redTexture = new THREE.TextureLoader().load('./assets/profile.jpg');
const profileCube = new THREE.Mesh(
  new THREE.BoxGeometry(3, 3, 3),
  new THREE.MeshBasicMaterial({ map: redTexture })
);
scene.add(profileCube);

function animate() {
  requestAnimationFrame(animate);

  torus.rotation.x += 0.01;
  torus.rotation.y += 0.005;
  torus.rotation.z += 0.01;

  //controls.update();

  renderer.render(scene, camera);
}

animate();