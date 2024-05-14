
export const register_user = async (formData) => {
    try {
        const res = await fetch('http://localhost:5000/api/Users', {
            headers: {
                'Content-Type': 'application/json',
            },
            method: 'POST',
            body: JSON.stringify(formData),
        });
        const data = res.json();
        return data;
    } catch (error) {
        console.log('Error in register_user (service) => ', error);
        return error.message
    }
};

export const login_user = async (formData) => {
    try {
        const res = await fetch('http://localhost:5000/api/Users/login', {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/plain'
            },
            method: 'POST',
            body: JSON.stringify(formData),
        });
        const data = await res.json();
        return data;
    } catch (error) {
        console.error('Error in login_user (service) => ', error);
        return { success: false, message: 'An error occurred while logging in.' };
    }
};


export const uploadFile = async (file, type,userId) => {
    const formData = new FormData();
    formData.append('formfile', file);
    formData.append('type', type);
    formData.append('userId', userId);

    try {
        const res = await fetch('http://localhost:5000/api/WorkContents', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        return data;
    } catch (error) {
        console.error('Error uploading file:', error);
        return { success: false, message: 'An error occurred while uploading file.' };
    }
};
