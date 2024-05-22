export const register_user = async (formData) => {
  try {
    const res = await fetch("http://localhost:5000/api/Users", {
      headers: {
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify(formData),
    });
    const data = res.json();
    return data;
  } catch (error) {
    console.log("Error in register_user (service) => ", error);
    return error.message;
  }
};

export const login_user = async (formData) => {
  try {
    const res = await fetch("http://localhost:5000/api/Users/login", {
      headers: {
        "Content-Type": "application/json",
        Accept: "text/plain",
      },
      method: "POST",
      body: JSON.stringify(formData),
    });
    const data = await res.json();
    return data;
  } catch (error) {
    console.error("Error in login_user (service) => ", error);
    return { success: false, message: "An error occurred while logging in." };
  }
};

export const uploadFile = async (file, type, doctorId, patientId) => {
  const formData = new FormData();
  formData.append("formfile", file);
  formData.append("type", type);
  formData.append("doctorId", doctorId);
  formData.append("patientId", patientId);

  console.log(patientId + doctorId);
  try {
    const res = await fetch("http://localhost:5000/api/WorkContents", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    return data;
  } catch (error) {
    console.error("Error uploading file:", error);
    return {
      success: false,
      message: "An error occurred while uploading file.",
    };
  }
};

export const checkPatientExistence = async (id) => {

  try {
    const myHeaders = new Headers();
    myHeaders.append("Accept", "text/plain");

    const requestOptions = {
      method: "GET",
      headers: myHeaders,
      redirect: "follow",
    };


    console.log("Ahmad Alawi 079 " + id);
    const response = await fetch(
      `http://localhost:5000/api/Users/${id}`,
      requestOptions
    );
    const data = await response.json();

    return data; // Resolve with the response data
  } catch (error) {
    console.error("Error checking patient existence:", error);
    throw error; // Re-throw the error
  }
};
