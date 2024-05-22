const BASE_URL = 'http://localhost:5000'; // Define the base URL of your API server

// Function to make a GET request to fetch all users
export const getUsers = async () => {
  try {
    const response = await fetch(`${BASE_URL}/api/Users`); // Use backticks for interpolation
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching users:', error.message);
    throw error;
  }
};

// Function to make a GET request to fetch a user by ID
export const getUserById = async (userId) => {
  try {
    const response = await fetch(`${BASE_URL}/api/Users/${userId}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching user by ID:', error.message);
    throw error;
  }
};

// Function to make a POST request to create a new user
export const createUser = async (userData) => {
  try {
    const response = await fetch(`${BASE_URL}/api/Users`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error creating user:', error.message);
    throw error;
  }
};

// Function to make a PUT request to update a user by ID
export const updateUser = async (userId, userData) => {
  try {
    const response = await fetch(`${BASE_URL}/api/Users/${userId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error updating user:', error.message);
    throw error;
  }
};

// Function to make a DELETE request to delete a user by ID
export const deleteUser = async (userId) => {
  try {
    const response = await fetch(`${BASE_URL}/api/Users/${userId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error deleting user:', error.message);
    throw error;
  }
};

// Function to make a POST request to login a user
export const loginUser = async (formData) => {
  try {
    const response = await fetch(`${BASE_URL}/api/Users/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error logging in:', error.message);
    throw error;
  }
};

// Function to make a GET request to fetch weather forecasts
export const getWeatherForecast = async () => {
  try {
    const response = await fetch('/WeatherForecast');
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching weather forecast:', error.message);
    throw error;
  }
};

// Function to make a GET request to fetch work contents by user ID
export const getWorkContentsByUserId = async (userId) => {
  try {
    const response = await fetch(`${BASE_URL}/api/WorkContents/user/${userId}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching work contents by user ID:', error.message);
    throw error;
  }
};

// Function to make a GET request to fetch all work contents
export const getWorkContents = async () => {
  try {
    const response = await fetch(`${BASE_URL}/api/WorkContents`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching work contents:', error.message);
    throw error;
  }
};

// Function to make a POST request to create a new work content
export const createWorkContent = async (workContentData) => {
  try {
    const response = await fetch(`${BASE_URL}/api/WorkContents`, {
      method: 'POST',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      body: JSON.stringify(workContentData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error creating work content:', error.message);
    throw error;
  }
};

// Function to make a PUT request to update a work content by ID
export const updateWorkContent = async (workContentId, workContentData) => {
  try {
    const response = await fetch(`${BASE_URL}/api/WorkContents/${workContentId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(workContentData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error updating work content:', error.message);
    throw error;
  }
};

// Function to make a DELETE request to delete a work content by ID
export const deleteWorkContent = async (workContentId) => {
  try {
    const response = await fetch(`${BASE_URL}/api/WorkContents/${workContentId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message);
    }
    return await response.json();
  } catch (error) {
    console.error('Error deleting work content:', error.message);
    throw error;
  }
};
