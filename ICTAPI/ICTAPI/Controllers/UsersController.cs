using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Linq;
using System.Threading.Tasks;
using System;
using System.ComponentModel.DataAnnotations;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using ICTAPI.ictDB;
using BCrypt.Net;
using Microsoft.IdentityModel.Tokens;

namespace ICTAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class UsersController : ControllerBase
    {
        private readonly IctAppContext _context;

        public UsersController(IctAppContext context)
        {
            _context = context;
        }

        // Define a common response structure
        public class ApiResponse
        {
            public bool Success { get; set; }
            public string Message { get; set; }
            public object Data { get; set; }
        }

        // GET: api/Users
        [HttpGet]
        public async Task<ActionResult<ApiResponse>> GetUsers()
        {
            var users = await _context.Users.ToListAsync();
            return Ok(new ApiResponse { Success = true, Message = "Users retrieved successfully", Data = users });
        }

        // GET: api/Users/5
        [HttpGet("{id}")]
        public async Task<ActionResult<ApiResponse>> GetUser(int id)
        {
            var user = await _context.Users.FindAsync(id);
            if (user == null)
            {
                return NotFound(new ApiResponse { Success = false, Message = "User not found", Data = null });
            }
            return Ok(new ApiResponse { Success = true, Message = "User retrieved successfully", Data = user });
        }

        // PUT: api/Users/5
        [HttpPut("{id}")]
        public async Task<IActionResult> PutUser(int id, User user)
        {
            if (id != user.Id)
            {
                return BadRequest(new ApiResponse { Success = false, Message = "Invalid user data", Data = null });
            }

            _context.Entry(user).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!UserExists(id))
                {
                    return NotFound(new ApiResponse { Success = false, Message = "User not found", Data = null });
                }
                else
                {
                    throw;
                }
            }

            return NoContent();
        }

        // POST: api/Users
        [HttpPost]
        public async Task<ActionResult<ApiResponse>> PostUser(User user)
        {
            if (user == null)
            {
                return BadRequest(new ApiResponse { Success = false, Message = "Invalid user data", Data = null });
            }

            var existingUser = await _context.Users.FirstOrDefaultAsync(u => u.Email == user.Email);
            if (existingUser != null)
            {
                return Conflict(new ApiResponse { Success = false, Message = $"User with email '{user.Email}' already exists", Data = null });
            }

            user.CreatedAt = DateTime.Now.ToString();
            var hashedPassword = BCrypt.Net.BCrypt.HashPassword(user.Password);
            user.Password = hashedPassword;

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return CreatedAtAction(nameof(GetUser), new { id = user.Id }, new ApiResponse { Success = true, Message = "User created successfully", Data = user });
        }

        // POST: api/Users/login
        [HttpPost("login")]
        public async Task<ActionResult<ApiResponse>> Login(LoginModel loginRequest)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == loginRequest.Email);

            if (user == null || !BCrypt.Net.BCrypt.Verify(loginRequest.Password, user.Password))
            {
                return NotFound(new ApiResponse { Success = false, Message = "Invalid credential", Data = null });
            }

            var tokenHandler = new JwtSecurityTokenHandler();
            var key = Encoding.ASCII.GetBytes("i@31866482uuryncyasd!@@@$$$Ggkdjfnv8448");//Encryption key 
            var tokenDescriptor = new SecurityTokenDescriptor
            {
                Subject = new ClaimsIdentity(new Claim[]
                {
                    new Claim(ClaimTypes.Name, user.Id.ToString())
                }),
                Expires = DateTime.UtcNow.AddDays(7),
                SigningCredentials = new SigningCredentials(new SymmetricSecurityKey(key), SecurityAlgorithms.HmacSha256Signature)
            };
            var token = tokenHandler.CreateToken(tokenDescriptor);
            var tokenString = tokenHandler.WriteToken(token);

            return Ok(new ApiResponse { Success = true, Message = "Login successful", Data = new { Token = tokenString, Id = user.Id ,fname = user.FName} });
        }

        private bool UserExists(int id)
        {
            return _context.Users.Any(e => e.Id == id);
        }
    }
}
