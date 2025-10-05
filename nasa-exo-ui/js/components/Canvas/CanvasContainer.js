import Exoplanets from "../Exoplanets/Exoplanets";

function CanvasContainer() {
  return (
    <Canvas camera={{ position: [0, 0, 500], fov: 60 }}>
      <ambientLight intensity={0.3} />
      <pointLight position={[100, 100, 100]} />
      <Exoplanets />   {/* <---- here’s your 3D planets */}
    </Canvas>
  );
}

export default CanvasContainer;
