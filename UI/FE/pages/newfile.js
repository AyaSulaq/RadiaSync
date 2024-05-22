import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import { uploadFile, checkPatientExistence } from "../services/index";
import Cookies from "js-cookie";
import { FaCheckCircle } from "react-icons/fa";

export default function NewFile() {
  const [fileType, setFileType] = useState("mri");
  const [duration, setDuration] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [patientExists, setPatientExists] = useState(false);
  const [patientId, setPatientId] = useState("");
  const [doctorId, setDoctorId] = useState("");
  const [borderColor, setBorderColor] = useState("border-gray-300");
  const [uploadMessage, setUploadMessage] = useState("");
  const router = useRouter();
  const userID = Cookies.get("userId");

  useEffect(() => {
    setDoctorId(userID);
  }, [userID]);

  const handleDurationChange = (e) => {
    setDuration(e.target.value);
    setPatientId(e.target.value); // Update the patientId as well
    setBorderColor("border-gray-300");
    setPatientExists(false);
  };

  const handleDurationBlur = async () => {
    if (patientId) {
      try {
        const res = await checkPatientExistence(patientId);
        setPatientExists(res.success);
        if (res.success) {
          setBorderColor("border-green-500");
        } else {
          setBorderColor("border-red-500");
        }
      } catch (error) {
        console.error("Error checking patient existence:", error);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Check if patientId is empty
    if (!patientId) {
      console.error('Patient ID is required');
      return;
    }

    // Check if selectedFile is empty
    if (!selectedFile) {
      console.error('File is required');
      return;
    }

    try {
      const res = await uploadFile(selectedFile, fileType, doctorId, patientId);
      if (res.success) {
        setUploadMessage("File uploaded successfully!");
        setTimeout(() => {
          router.push('/home');
        }, 2000);
      } else {
        console.error('Failed to upload file');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div className="w-full h-screen flex items-center justify-center bg-blue-500">
      <div className="bg-white p-8 rounded-lg shadow-md">
        <h2 className="text-2xl font-bold mb-4">Upload New File</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="fileType" className="block font-semibold mb-2">
              File Type
            </label>
            <div>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="fileType"
                  value="mri"
                  checked={fileType === "mri"}
                  onChange={() => setFileType("mri")}
                  className="mr-2"
                />
                MRI
              </label>
              <label className="inline-flex items-center ml-4">
                <input
                  type="radio"
                  name="fileType"
                  value="ct"
                  checked={fileType === "ct"}
                  onChange={() => setFileType("ct")}
                  className="mr-2"
                />
                CT
              </label>
            </div>
          </div>
          <div className="mb-4 relative">
            <label htmlFor="duration" className="block font-semibold mb-2">
              Patient
            </label>
            <input
              type="text"
              id="duration"
              value={duration}
              onChange={handleDurationChange}
              onBlur={handleDurationBlur}
              className={`border p-2 w-full ${borderColor}`}
            />
            {patientExists && (
              <FaCheckCircle className="absolute right-2 top-10 text-green-500" />
            )}
          </div>
          <div className="mb-4">
            <label htmlFor="file" className="block font-semibold mb-2">
              Browse File
            </label>
            <input
              type="file"
              id="file"
              onChange={(e) => setSelectedFile(e.target.files[0])}
              className="border p-2 w-full"
            />
          </div>
          <button
            type="submit"
            className="bg-blue-500 text-white font-semibold py-2 px-4 rounded hover:bg-blue-600"
          >
            Upload
          </button>
          {uploadMessage && (
            <p className="mt-4 text-green-500 font-semibold">{uploadMessage}</p>
          )}
        </form>
      </div>
    </div>
  );
}
