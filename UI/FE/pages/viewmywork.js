import React, { useState, useEffect } from "react";
import moment from "moment";
import Modal from "./Modal";
import { ToastContainer, toast } from "react-toastify";
import Cookies from "js-cookie";
import Router from "next/router";

export default function ViewAllWork() {
  const [userId, setUserId] = useState(null);
  const [workContents, setWorkContents] = useState([]);
  const [modalImages, setModalImages] = useState([]);
  const [openModal, setOpenModal] = useState(false);

  const fName = Cookies.get("fName");

  useEffect(() => {
    const userIdFromCookie = Cookies.get("userId");
    if (userIdFromCookie) {
      setUserId(userIdFromCookie);
    }
  }, []);

  useEffect(() => {
    if (userId) {
      fetchWorkContents(userId);
    }
  }, [userId]);

  const fetchWorkContents = async (userId) => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/WorkContents/byUser/${userId}`,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        }
      );
      const data = await response.json();
      if (data.success) {
        setWorkContents(data.data);
      } else {
        toast.error(data.message);
      }
    } catch (error) {
      console.error("Error fetching work contents:", error);
    }
  };

  const handleViewResult = async (id) => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/WorkContents/${id}`,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        }
      );

      const data = await response.json();
      if (data.success) {
        const imageUrls = data.data;
        setModalImages(imageUrls);
        setOpenModal(true);
      } else {
        console.error("Failed to fetch images:", data.message);
      }
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  const handleCloseModal = () => {
    setOpenModal(false);
  };

  const handleBack = () => {
    Router.push('/home');
  };

  return (
    <div>
      <div className="flex justify-start items-center p-4">
        <button 
          onClick={handleBack}
          className="bg-white border-2 border-white hover:bg-transparent transition-all text-indigo-700 hover:text-white font-semibold text-lg px-4 py-2 rounded duration-700">
          Back
        </button>
      </div>

      <div className="flex justify-center items-center ">
        <p className="text-4xl font-extrabold mb-4">
          Welcome {fName} to your History
        </p>
      </div>

      <div
        className="container mx-auto mt-8"
        style={{ background: "rgb(67, 56, 202)" }}
      >
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-blue-200">
            <thead>
              <tr className="bg-blue-100 border-b border-blue-200">
                <th className="px-6 py-3 text-left text-xs font-semibold text-blue-600 uppercase tracking-wider">
                  File Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-blue-600 uppercase tracking-wider">
                  Creation Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-blue-600 uppercase tracking-wider">
                  Finished Date
                </th>
                <th className="px-6 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-blue-200">
              {workContents.map((workContent) => (
                <tr key={workContent.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {workContent.fileName}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {moment(workContent.createdAt).format(
                      "YYYY-MM-DD HH:mm:ss"
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {workContent.finishedAt
                      ? moment(workContent.finishedAt).format(
                          "YYYY-MM-DD HH:mm:ss"
                        )
                      : "N/A"}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <button
                      className="text-blue-600 hover:text-blue-800 font-semibold mr-4"
                      onClick={() => handleViewResult(workContent.id)}
                    >
                      View Result
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Modal open={openModal} onClose={handleCloseModal} images={modalImages} />
    </div>
  );
}
