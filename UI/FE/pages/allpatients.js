import React, { useEffect, useContext, useState } from "react";
import Cookies from "js-cookie";
import Router from "next/router";
import axios from "axios";
import { UserContext } from "../services/UserContext";
import styles from "./AllPatients.module.css"; // Import CSS module for animation

export default function AllPatients() {
  const [patients, setPatients] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const token = Cookies.get("token");
  const { fName, setFName, userId } = useContext(UserContext);

  useEffect(() => {
    if (!token) {
      Router.push("/");
    } else {
      fetchPatients(currentPage);
    }
  }, [currentPage]);

  useEffect(() => {
    const interval = setInterval(() => {
      generateRandomCard();
    }, 3000); // Adjust the interval as needed

    return () => clearInterval(interval);
  }, []);

  const fetchPatients = async (page) => {
    try {
      const response = await axios.get(
        `http://localhost:5000/api/Users?type=patient&pageNumber=${page}&pageSize=30`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const { data } = response.data;
      setPatients(data.users);
      setTotalPages(data.pagination.totalPages);
    } catch (error) {
      console.error("Error fetching patients:", error);
    }
  };

  const handleNextPage = () => {
    setCurrentPage(currentPage + 1);
  };

  const handlePrevPage = () => {
    setCurrentPage(currentPage - 1);
  };

  const generateRandomCard = async () => {
    try {
      const response = await axios.get(
        `http://localhost:5000/api/Users?type=patient&pageSize=1`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const { data } = response.data;
      const randomPatient = data.users[0];
      const cardElement = (
        <div
          key={randomPatient.id}
          className={styles["rain-card"]}
          style={{
            top: `${Math.random() * 100}vh`,
            left: `${Math.random() * 100}vw`,
          }}
        >
          <p>
            Name: {randomPatient.fName} {randomPatient.lName}
          </p>
          <p>Email: {randomPatient.email}</p>
          <p>ID: {randomPatient.id}</p>
        </div>
      );
      setPatients((prevPatients) => [...prevPatients, cardElement]);
    } catch (error) {
      console.error("Error generating random card:", error);
    }
  };

  return (
    <div className="w-full h-screen flex flex-col items-center justify-center text-white bg-indigo-700 tracking-widest uppercase">
      <h1 className="text-4xl font-extrabold mb-4">All Patients</h1>
      <div className={styles["rain-animation-container"]}>{patients}</div>
      <div className="flex mt-4">
        <button onClick={handlePrevPage} disabled={currentPage === 1}>
          Previous
        </button>
        <button onClick={handleNextPage} disabled={currentPage === totalPages}>
          Next
        </button>
      </div>
    </div>
  );
}
