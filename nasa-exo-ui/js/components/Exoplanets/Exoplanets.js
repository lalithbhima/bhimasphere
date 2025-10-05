import { useMemo } from "react";
import { exoplanets } from "../../data/exoplanets";

function Exoplanets() {
  // Memoize so the dataset isn’t recalculated on each render
  const planets = useMemo(() => exoplanets, []);

  return (
    <>
      {planets.map((planet, i) => {
        // Scale radius based on size & probability
        const radius = Math.log10(planet.rade_Re + 2) * 0.2;
        const color =
          planet.pred_text === "Confirmed"
            ? "limegreen"
            : planet.pred_text === "Candidate"
            ? "gold"
            : "red";

        return (
          <mesh key={i} position={[planet.x, planet.y, planet.z]}>
            <sphereGeometry args={[radius, 16, 16]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={0.5}
            />
          </mesh>
        );
      })}
    </>
  );
}

export default Exoplanets;
