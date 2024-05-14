using Google.Protobuf.WellKnownTypes;
using Microsoft.AspNetCore.Builder;
using ICTAPI;
using ICTAPI.ictDB;
using Pomelo.EntityFrameworkCore.MySql; // Add this line
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
// Add anti-forgery services
builder.Services.AddAntiforgery(options =>
{
    // Set options here if needed
});

builder.Services.AddDbContext<IctAppContext>(options =>
    options.UseMySQL("server=74.208.107.56;uid=ictApp;pwd=ictApp123;database=ictApp"));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

//app.UseHttpsRedirection();
//app.UseAuthorization();

app.MapControllers();


// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

// Enable routing
app.UseRouting();

// Add CORS middleware
app.UseCors(builder =>
{
    builder.AllowAnyOrigin()
           .AllowAnyMethod()
           .AllowAnyHeader();
});

// Enable authentication and authorization
//app.UseAuthentication();
//app.UseAuthorization();

// Map controllers
app.UseEndpoints(endpoints =>
{
    endpoints.MapControllers();
});



app.Run();
