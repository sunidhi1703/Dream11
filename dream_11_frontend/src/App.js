import React, { useState } from "react";
import styled, { keyframes } from "styled-components";


const Container = styled.div`
  max-width: 900px;
  margin: 40px auto;
  font-family: "Inter", sans-serif;
  color: #222;
  padding: 0 20px;
`;

const Header = styled.h1`
  text-align: center;
  color: #1b5e20;
  font-size: 2.2rem;
  margin-bottom: 8px;
`;

const Subtext = styled.p`
  text-align: center;
  color: #555;
  margin-bottom: 30px;
`;

const UploadSection = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
`;

const FileInput = styled.input`
  background: ${(props) => (props.disabled ? "#bdbdbd" : "#43a047")};
  color: white;
  border: none;
  padding: 6px;
  border-radius: 6px;
  font-weight: 600;
  width: 200px;
`;

const Button = styled.button`
  background: ${(props) => (props.disabled ? "#bdbdbd" : "#43a047")};
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 6px;
  cursor: ${(props) => (props.disabled ? "not-allowed" : "pointer")};
  font-weight: 600;
  transition: background 0.3s ease;
  &:hover {
    background: ${(props) => (props.disabled ? "#bdbdbd" : "#2e7d32")};
  }
`;

const ErrorMsg = styled.p`
  color: #d32f2f;
  text-align: center;
  font-weight: 500;
`;

const LoaderText = styled.p`
  text-align: center;
  color: #555;
`;

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
`;

const TeamCard = styled.div`
  animation: ${fadeIn} 0.4s ease;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 28px;
  background: #fafafa;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
`;

const TeamTitle = styled.h3`
  color: #1b5e20;
  font-size: 1.4rem;
  margin-bottom: 10px;
`;

const Stats = styled.p`
  margin-bottom: 10px;
  font-weight: 600;
  color: #333;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
`;
const TabsRow = styled.div`
display: flex;
flex-direction: row;
gap: 20px;
`;
const Th = styled.th`
  padding: 10px;
  text-align: left;
  border-bottom: 2px solid #c8e6c9;
  color: #2e7d32;
`;

const Td = styled.td`
  padding: 8px 10px;
  border-bottom: 1px solid #eee;
  text-align: ${(props) => props.align || "left"};
`;

const FinalScore = styled.div`
  text-align: center;
  margin-top: 25px;
`;

const ScoreValue = styled.p`
  font-size: 2.4rem;
  font-weight: 700;
  color: #2e7d32;
  margin: 0;
`;

// --- Team Table Component ---

const TeamTable = ({ title, team, pointColumnName }) => {
  if (!team || team.length === 0) {
    return <h4>{title}</h4>;
  }

  const totalPoints = team.reduce(
    (sum, player) => sum + (player[pointColumnName] || 0),
    0
  );
  const totalCredits = team.reduce(
    (sum, player) => sum + (player.credits || 0),
    0
  );

  return (
    <TeamCard>
      <TeamTitle>{title}</TeamTitle>
      <Stats>
        Total Points: {totalPoints.toFixed(2)} | Total Credits:{" "}
        {totalCredits.toFixed(2)} / 100.0
      </Stats>
      <Table>
        <thead>
          <tr>
            <Th>Player</Th>
            <Th align="center">Role</Th>
            <Th align="right">Credits</Th>
            <Th align="right">Points</Th>
          </tr>
        </thead>
        <tbody>
          {team.map((player, index) => (
            <tr key={index}>
              <Td>{player.player_name} ({player.team})</Td>
              <Td align="center">{player.role}</Td>
              <Td align="right">{player.credits.toFixed(2)}</Td>
              <Td align="right" style={{ fontWeight: "bold" }}>
                {(player[pointColumnName] || 0).toFixed(2)}
              </Td>
            </tr>
          ))}
        </tbody>
      </Table>
    </TeamCard>
  );
};


export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [predictionData, setPredictionData] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setError("");
    setPredictionData(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select a JSON file first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setPredictionData(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5001/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Something went wrong with the backend.");
      }

      const data = await response.json();
      setPredictionData(data);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container>
      <Header>Dream11 AI Predictor</Header>
      <Subtext>Upload a match JSON file to get the predicted and dream teams.</Subtext>

      <UploadSection>
        <FileInput type="file" accept=".json" onChange={handleFileChange} />
        <Button onClick={handleSubmit} disabled={isLoading}>
          {isLoading ? "Processing..." : "Get Prediction"}
        </Button>
      </UploadSection>

      {isLoading && <LoaderText>Running AI model... please wait.</LoaderText>}
      {error && <ErrorMsg>Error: {error}</ErrorMsg>}

      {predictionData && (
        <>
          <TabsRow>
            <TeamTable
              title="Recommended XI (Predicted)"
              team={predictionData.predictedXI}
              pointColumnName="predicted_fp"
            />
            <TeamTable
              title="Official Dream XI (Actual)"
              team={predictionData.dreamXI}
              pointColumnName="actual_fp"
            />
          </TabsRow>
        </>
      )}
    </Container>
  );
}
